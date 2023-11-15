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
package org.springframework.ai.vectorstore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.http.MediaType;
import org.springframework.util.Assert;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.util.UriComponentsBuilder;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * A VectorStore implementation backed by GemFire. This store supports creating, updating,
 * deleting, and similarity searching of documents in a GemFire index.
 *
 * @author Anthony Baker
 */
public class GemFireVectorStore implements VectorStore {

	private static final String DISTANCE_METADATA_FIELD_NAME = "distance";

	private final WebClient client;

	private final EmbeddingClient embeddingClient;

	private final int topKPerBucket;

	private final String documentField;

	public static final class GemFireVectorStoreConfig {

		private final WebClient client;

		private final int topKPerBucket;

		private final String documentField;

		public static Builder builder() {
			return new Builder();
		}

		private GemFireVectorStoreConfig(Builder builder) {
			String base = UriComponentsBuilder.fromUriString("http{ssl}://{host}:{port}/gemfire-vectordb/v1/{index}")
				.build(builder.sslEnabled ? "s" : "", builder.host, builder.port, builder.index)
				.toString();

			this.client = WebClient.create(base);
			this.topKPerBucket = builder.topKPerBucket;
			this.documentField = builder.documentField;
		}

		public static class Builder {

			private String host;

			private int port = DEFAULT_PORT;

			private boolean sslEnabled;

			private long connectionTimeout;

			private long requestTimeout;

			private String index;

			private int topKPerBucket = DEFAULT_TOP_K_PER_BUCKET;

			private String documentField = DEFAULT_DOCUMENT_FIELD;

			public Builder withHost(String host) {
				Assert.hasText(host, "host must have a value");
				this.host = host;
				return this;
			}

			public Builder withPort(int port) {
				Assert.isTrue(port > 0, "port must be postive");
				this.port = port;
				return this;
			}

			public Builder withSslEnabled(boolean sslEnabled) {
				this.sslEnabled = sslEnabled;
				return this;
			}

			public Builder withConnectionTimeout(long timeout) {
				Assert.isTrue(timeout >= 0, "timeout must be >= 0");
				this.connectionTimeout = timeout;
				return this;
			}

			public Builder withRequestTimeout(long timeout) {
				Assert.isTrue(timeout >= 0, "timeout must be >= 0");
				this.requestTimeout = timeout;
				return this;
			}

			public Builder withIndex(String index) {
				Assert.hasText(index, "index must have a value");
				this.index = index;
				return this;
			}

			public Builder withTopKPerBucket(int topKPerBucket) {
				Assert.isTrue(topKPerBucket > 0, "topKPerBucket must be positive");
				this.topKPerBucket = topKPerBucket;
				return this;
			}

			public Builder withDocumentField(String documentField) {
				Assert.hasText(documentField, "documentField must have a value");
				this.documentField = documentField;
				return this;
			}

			public GemFireVectorStoreConfig build() {
				return new GemFireVectorStoreConfig(this);
			}

		}

	}

	private static final int DEFAULT_PORT = 8080;

	private static final int DEFAULT_TOP_K_PER_BUCKET = 50;

	private static final String DEFAULT_DOCUMENT_FIELD = "document";

	public GemFireVectorStore(GemFireVectorStoreConfig config, EmbeddingClient embedding) {
		Assert.notNull(config, "GemFireVectorStoreConfig must not be null");
		Assert.notNull(embedding, "EmbeddingClient must not be null");

		this.client = config.client;
		this.embeddingClient = embedding;

		this.topKPerBucket = config.topKPerBucket;
		this.documentField = config.documentField;
	}

	private static final class CreateRequest {

		@JsonProperty("embedding-type")
		private final String embeddingType = "float";

	}

	private static final class UploadRequest {

		private static final class Embedding {

			private final String key;

			private List<Double> vector;

			private Map<String, Object> metadata;

			public Embedding(String key, List<Double> vector, String contentName, String content,
					Map<String, Object> metadata) {
				this.key = key;
				this.vector = vector;
				this.metadata = new HashMap<>(metadata);
				this.metadata.put(contentName, content);
			}

			public String getKey() {
				return key;
			}

			public List<Double> getVector() {
				return vector;
			}

			public Map<String, Object> getMetadata() {
				return metadata;
			}

		}

		private final List<Embedding> embeddings;

		public List<Embedding> getEmbeddings() {
			return embeddings;
		}

		public UploadRequest(List<Embedding> embeddings) {
			this.embeddings = embeddings;
		}

	}

	private static final class QueryRequest {

		private final List<Double> vector;

		private final int k;

		private final int kPerBucket;

		private final boolean includeMetadata;

		public QueryRequest(List<Double> vector, int k, int kPerBucket, boolean includeMetadata) {
			this.vector = vector;
			this.k = k;
			this.kPerBucket = kPerBucket;
			this.includeMetadata = includeMetadata;
		}

		@JsonProperty("embedding")
		public List<Double> getVector() {
			return vector;
		}

		public int getK() {
			return k;
		}

		@JsonProperty("k-per-bucket")
		public int getkPerBucket() {
			return kPerBucket;
		}

		@JsonProperty("include-metadata")
		public boolean isIncludeMetadata() {
			return includeMetadata;
		}

	}

	private static final class QueryResponse {

		private String key;

		private float score;

		private Map<String, Object> metadata;

		private String getContent(String field) {
			return (String) metadata.get(field);
		}

		public void setKey(String key) {
			this.key = key;
		}

		public void setScore(float score) {
			this.score = score;
		}

		public void setMetadata(Map<String, Object> metadata) {
			this.metadata = metadata;
		}

	}

	@Override
	public void add(List<Document> documents) {
		UploadRequest upload = new UploadRequest(documents.stream().map(document -> {
			// Compute and assign an embedding to the document.
			document.setEmbedding(this.embeddingClient.embed(document));
			return new UploadRequest.Embedding(document.getId(), document.getEmbedding(), documentField,
					document.getContent(), document.getMetadata());
		}).toList());

		client.put()
			.uri("/keys")
			.contentType(MediaType.APPLICATION_JSON)
			.bodyValue(upload)
			.retrieve()
			.bodyToMono(Void.class)
			.block();
	}

	@Override
	public Optional<Boolean> delete(List<String> idList) {
		throw new UnsupportedOperationException("This vector store doesn't support delete (yet)");
	}

	@Override
	public List<Document> similaritySearch(SearchRequest request) {
		List<Double> vector = this.embeddingClient.embed(request.getQuery());

		return client.post()
			.uri("/query")
			.contentType(MediaType.APPLICATION_JSON)
			.bodyValue(new QueryRequest(vector, request.getTopK(), topKPerBucket, true))
			.retrieve()
			.bodyToFlux(QueryResponse.class)
			.filter(r -> r.score >= request.getSimilarityThreshold())
			.map(r -> {
				Map<String, Object> metadata = r.metadata;
				metadata.put(DISTANCE_METADATA_FIELD_NAME, 1 - r.score);
				String content = (String) metadata.remove(documentField);

				return new Document(r.key, content, metadata);
			})
			.collectList()
			.block();
	}

	public void createIndex() {
		client.post()
			.contentType(MediaType.APPLICATION_JSON)
			.bodyValue(new CreateRequest())
			.retrieve()
			.bodyToMono(Void.class)
			.block();
	}

	public void deleteIndex() {
		client.delete().uri("?delete-data=true").retrieve().bodyToMono(Void.class).block();
	}

}
