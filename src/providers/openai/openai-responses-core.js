import axios from 'axios';
import logger from '../../utils/logger.js';
import { existsSync, readFileSync } from 'fs';
import * as http from 'http';
import * as https from 'https';
import { configureAxiosProxy, configureTLSSidecar, isTLSSidecarEnabledForProvider } from '../../utils/proxy-utils.js';
import { MODEL_PROVIDER, getRetryAfterMs } from '../../utils/common.js';
import { normalizeModelIds } from '../provider-models.js';

function getProviderType(config) {
    const providerType = config?.MODEL_PROVIDER;
    if (typeof providerType === 'string' &&
        (providerType === MODEL_PROVIDER.OPENAI_CUSTOM_RESPONSES || providerType.startsWith(MODEL_PROVIDER.OPENAI_CUSTOM_RESPONSES + '-'))) {
        return providerType;
    }

    return MODEL_PROVIDER.OPENAI_CUSTOM_RESPONSES;
}

function getManagedModelsFromPools(config, providerType) {
    const pools = config?.providerPools;
    const providerEntries = Array.isArray(pools?.[providerType]) ? pools[providerType] : [];
    const poolModels = providerEntries.flatMap(provider => Array.isArray(provider?.supportedModels) ? provider.supportedModels : []);

    if (poolModels.length > 0) {
        return normalizeModelIds(poolModels);
    }

    const filePath = config?.PROVIDER_POOLS_FILE_PATH || 'configs/provider_pools.json';
    if (!existsSync(filePath)) {
        return [];
    }

    try {
        const filePools = JSON.parse(readFileSync(filePath, 'utf-8'));
        const fileEntries = Array.isArray(filePools?.[providerType]) ? filePools[providerType] : [];
        const fileModels = fileEntries.flatMap(provider => Array.isArray(provider?.supportedModels) ? provider.supportedModels : []);
        return normalizeModelIds(fileModels);
    } catch (error) {
        logger.warn(`[OpenAIResponses] Failed to load provider pool models from ${filePath}: ${error.message}`);
        return [];
    }
}

// OpenAI Responses API specification service for interacting with third-party models
export class OpenAIResponsesApiService {
    constructor(config) {
        if (!config.OPENAI_API_KEY) {
            throw new Error("OpenAI API Key is required for OpenAIResponsesApiService.");
        }
        this.config = config;
        this.apiKey = config.OPENAI_API_KEY;
        this.baseUrl = config.OPENAI_BASE_URL || 'https://api.openai.com/v1';
        this.useSystemProxy = config?.USE_SYSTEM_PROXY_OPENAI ?? false;
        logger.info(`[OpenAIResponses] System proxy ${this.useSystemProxy ? 'enabled' : 'disabled'}`);

        // 配置 HTTP/HTTPS agent 限制连接池大小，避免资源泄漏
        const httpAgent = new http.Agent({
            keepAlive: true,
            maxSockets: 100,
            maxFreeSockets: 5,
            timeout: 120000,
        });
        const httpsAgent = new https.Agent({
            keepAlive: true,
            maxSockets: 100,
            maxFreeSockets: 5,
            timeout: 120000,
        });

        // 检查是否启用了 TLS Sidecar
        const isTLSSidecarEnabled = isTLSSidecarEnabledForProvider(config, config.MODEL_PROVIDER || MODEL_PROVIDER.OPENAI_CUSTOM_RESPONSES);

        const axiosConfig = {
            baseURL: this.baseUrl,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiKey}`
            }
        };

        // 如果启用了 TLS Sidecar，就不配置 httpAgent 和 httpsAgent，避免配置冲突
        if (!isTLSSidecarEnabled) {
            axiosConfig.httpAgent = httpAgent;
            axiosConfig.httpsAgent = httpsAgent;
            // 配置自定义代理 (使用 openai-custom 的代理配置)
            configureAxiosProxy(axiosConfig, config, config.MODEL_PROVIDER || MODEL_PROVIDER.OPENAI_CUSTOM_RESPONSES);
        }

        this.axiosInstance = axios.create(axiosConfig);

    }

    getAllowedModels() {
        const providerType = getProviderType(this.config);
        const directModels = normalizeModelIds(this.config?.supportedModels);
        if (directModels.length > 0) {
            return directModels;
        }

        return getManagedModelsFromPools(this.config, providerType);
    }

    ensureModelIsAllowed(model) {
        const allowedModels = this.getAllowedModels();
        if (allowedModels.length > 0 && model && !allowedModels.includes(model)) {
            const error = new Error(`Model '${model}' is not enabled for ${getProviderType(this.config)}. Allowed models: ${allowedModels.join(', ')}`);
            error.status = 400;
            throw error;
        }
    }

    _applySidecar(axiosConfig) {
        return configureTLSSidecar(axiosConfig, this.config, this.config.MODEL_PROVIDER || MODEL_PROVIDER.OPENAI_CUSTOM_RESPONSES, this.baseUrl);
    }

    async callApi(endpoint, body, isRetry = false, retryCount = 0) {
        const maxRetries = this.config.REQUEST_MAX_RETRIES || 3;
        const baseDelay = this.config.REQUEST_BASE_DELAY || 1000;  // 1 second base delay

        try {
            const axiosConfig = {
                method: 'post',
                url: endpoint,
                data: body
            };
            this._applySidecar(axiosConfig);
            const response = await this.axiosInstance.request(axiosConfig);
            return response.data;
        } catch (error) {
            const status = error.response?.status;
            const data = error.response?.data;
            if (status === 401 || status === 403) {
                logger.error(`[API] Received ${status}. API Key might be invalid or expired.`);
                throw error;
            }

            // Handle 429 (Too Many Requests)
            if (status === 429) {
                const retryAfter = getRetryAfterMs(error);
                if (retryAfter !== null) {
                    logger.warn(`[API] Received 429 with Retry-After: ${retryAfter}ms. Throwing to upper layer.`);
                    throw error;
                }
                if (retryCount < maxRetries) {
                    const delay = baseDelay * Math.pow(2, retryCount);
                    logger.info(`[API] Received 429 (Too Many Requests). No Retry-After found. Retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries})`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    return this.callApi(endpoint, body, isRetry, retryCount + 1);
                }
            }

            // Handle other retryable errors (5xx server errors)
            if (status >= 500 && status < 600 && retryCount < maxRetries) {
                const delay = baseDelay * Math.pow(2, retryCount);
                logger.info(`[API] Received ${status} server error. Retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.callApi(endpoint, body, isRetry, retryCount + 1);
            }

            logger.error(`Error calling OpenAI Responses API (Status: ${status}):`, error.message);
            throw error;
        }
    }

    async *streamApi(endpoint, body, isRetry = false, retryCount = 0) {
        const maxRetries = this.config.REQUEST_MAX_RETRIES || 3;
        const baseDelay = this.config.REQUEST_BASE_DELAY || 1000;  // 1 second base delay

        // OpenAI 的流式请求需要将 stream 设置为 true
        const streamRequestBody = { ...body, stream: true };

        try {
            const axiosConfig = {
                method: 'post',
                url: endpoint,
                data: streamRequestBody,
                responseType: 'stream'
            };
            this._applySidecar(axiosConfig);
            const response = await this.axiosInstance.request(axiosConfig);

            const stream = response.data;
            let buffer = '';

            for await (const chunk of stream) {
                buffer += chunk.toString();
                let newlineIndex;
                while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                    const line = buffer.substring(0, newlineIndex).trim();
                    buffer = buffer.substring(newlineIndex + 1);

                    if (line.startsWith('data: ')) {
                        const jsonData = line.substring(6).trim();
                        if (jsonData === '[DONE]') {
                            return; // Stream finished
                        }
                        try {
                            const parsedChunk = JSON.parse(jsonData);
                            yield parsedChunk;
                        } catch (e) {
                            logger.warn("[OpenAIResponsesApiService] Failed to parse stream chunk JSON:", e.message, "Data:", jsonData);
                        }
                    } else if (line === '') {
                        // Empty line, end of an event
                    }
                }
            }
        } catch (error) {
            const status = error.response?.status;
            const data = error.response?.data;
            if (status === 401 || status === 403) {
                logger.error(`[API] Received ${status} during stream. API Key might be invalid or expired.`);
                throw error;
            }

            // Handle 429 (Too Many Requests)
            if (status === 429) {
                const retryAfter = getRetryAfterMs(error);
                if (retryAfter !== null) {
                    logger.warn(`[API] Received 429 with Retry-After: ${retryAfter}ms during stream. Throwing to upper layer.`);
                    throw error;
                }
                if (retryCount < maxRetries) {
                    const delay = baseDelay * Math.pow(2, retryCount);
                    logger.info(`[API] Received 429 (Too Many Requests) during stream. No Retry-After found. Retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries})`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    yield* this.streamApi(endpoint, body, isRetry, retryCount + 1);
                    return;
                }
            }

            // Handle other retryable errors (5xx server errors)
            if (status >= 500 && status < 600 && retryCount < maxRetries) {
                const delay = baseDelay * Math.pow(2, retryCount);
                logger.info(`[API] Received ${status} server error during stream. Retrying in ${delay}ms... (attempt ${retryCount + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                yield* this.streamApi(endpoint, body, isRetry, retryCount + 1);
                return;
            }

            logger.error(`Error calling OpenAI Responses streaming API (Status: ${status}):`, error.message);
            throw error;
        }
    }

    async generateContent(model, requestBody) {
        this.ensureModelIsAllowed(model);

        // 临时存储 monitorRequestId
        if (requestBody._monitorRequestId) {
            this.config._monitorRequestId = requestBody._monitorRequestId;
            delete requestBody._monitorRequestId;
        }
        if (requestBody._requestBaseUrl) {
            delete requestBody._requestBaseUrl;
        }

        return this.callApi('/responses', requestBody);
    }

    async *generateContentStream(model, requestBody) {
        this.ensureModelIsAllowed(model);

        // 临时存储 monitorRequestId
        if (requestBody._monitorRequestId) {
            this.config._monitorRequestId = requestBody._monitorRequestId;
            delete requestBody._monitorRequestId;
        }
        if (requestBody._requestBaseUrl) {
            delete requestBody._requestBaseUrl;
        }

        yield* this.streamApi('/responses', requestBody);
    }

    async listModels() {
        const allowedModels = this.getAllowedModels();
        if (allowedModels.length > 0) {
            return {
                object: 'list',
                data: allowedModels.map(modelId => ({
                    id: modelId,
                    object: 'model',
                    created: Math.floor(Date.now() / 1000),
                    owned_by: getProviderType(this.config)
                }))
            };
        }

        try {
            const axiosConfig = {
                method: 'get',
                url: '/models'
            };
            this._applySidecar(axiosConfig);
            const response = await this.axiosInstance.request(axiosConfig);
            return response.data;
        } catch (error) {
            const status = error.response?.status;
            const data = error.response?.data;
            logger.error(`Error listing OpenAI Responses models (Status: ${status}):`, data || error.message);
            throw error;
        }
    }
}
