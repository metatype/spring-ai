/*
 * Copyright 2023-2023 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.autoconfigure.vectorstore.milvus;

import java.util.concurrent.TimeUnit;

import io.milvus.client.MilvusServiceClient;
import io.milvus.param.ConnectParam;

import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.vectorstore.MilvusVectorStore;
import org.springframework.ai.vectorstore.MilvusVectorStore.MilvusVectorStoreConfig;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/**
 * @author Christian Tzolov
 */
@AutoConfiguration
@ConditionalOnClass({ MilvusVectorStore.class, EmbeddingClient.class })
@EnableConfigurationProperties({ MilvusServiceClientProperties.class, MilvusVectorStoreProperties.class })
public class MilvusVectorStoreAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	public VectorStore vectorStore(MilvusServiceClient milvusClient, EmbeddingClient embeddingClient,
			MilvusVectorStoreProperties properties) {

		MilvusVectorStoreConfig config = MilvusVectorStoreConfig.builder()
			.withCollectionName(properties.getCollectionName())
			.withDatabaseName(properties.getDatabaseName())
			.withIndexType(properties.getIndexType())
			.withMetricType(properties.getMetricType())
			.withIndexParameters(properties.getIndexParameters())
			.build();

		return new MilvusVectorStore(milvusClient, embeddingClient, config);
	}

	@Bean
	@ConditionalOnMissingBean
	public MilvusServiceClient milvusClient(MilvusVectorStoreProperties serverProperties,
			MilvusServiceClientProperties clientProperties) {

		var builder = ConnectParam.newBuilder()
			.withHost(clientProperties.getHost())
			.withPort(clientProperties.getPort())
			.withDatabaseName(serverProperties.getDatabaseName())
			.withConnectTimeout(clientProperties.getConnectTimeoutMs(), TimeUnit.MILLISECONDS)
			.withKeepAliveTime(clientProperties.getKeepAliveTimeMs(), TimeUnit.MILLISECONDS)
			.withKeepAliveTimeout(clientProperties.getKeepAliveTimeoutMs(), TimeUnit.MILLISECONDS)
			.withRpcDeadline(clientProperties.getRpcDeadlineMs(), TimeUnit.MILLISECONDS)
			.withSecure(clientProperties.isSecure())
			.withIdleTimeout(clientProperties.getIdleTimeoutMs(), TimeUnit.MILLISECONDS)
			.withAuthorization(clientProperties.getUsername(), clientProperties.getPassword());

		if (clientProperties.isSecure() && StringUtils.hasText(clientProperties.getUri())) {
			builder.withUri(clientProperties.getUri());
		}

		if (clientProperties.isSecure() && StringUtils.hasText(clientProperties.getToken())) {
			builder.withUri(clientProperties.getToken());
		}

		if (clientProperties.isSecure() && StringUtils.hasText(clientProperties.getClientKeyPath())) {
			builder.withUri(clientProperties.getClientKeyPath());
		}

		if (clientProperties.isSecure() && StringUtils.hasText(clientProperties.getClientPemPath())) {
			builder.withUri(clientProperties.getClientPemPath());
		}

		if (clientProperties.isSecure() && StringUtils.hasText(clientProperties.getCaPemPath())) {
			builder.withUri(clientProperties.getCaPemPath());
		}

		if (clientProperties.isSecure() && StringUtils.hasText(clientProperties.getServerPemPath())) {
			builder.withUri(clientProperties.getServerPemPath());
		}

		if (clientProperties.isSecure() && StringUtils.hasText(clientProperties.getServerName())) {
			builder.withUri(clientProperties.getServerName());
		}

		return new MilvusServiceClient(builder.build());
	}

}
