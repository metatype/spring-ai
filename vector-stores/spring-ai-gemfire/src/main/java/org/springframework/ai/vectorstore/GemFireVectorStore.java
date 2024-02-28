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

import static org.springframework.http.HttpStatus.BAD_REQUEST;
import static org.springframework.http.HttpStatus.CONFLICT;
import static org.springframework.http.HttpStatus.NOT_FOUND;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ClientHttpRequest;
import org.springframework.util.Assert;
import org.springframework.web.reactive.function.BodyInserter;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientException;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.util.UriComponentsBuilder;

import com.fasterxml.jackson.annotation.JsonProperty;
import reactor.util.annotation.NonNull;

/**
 * A VectorStore implementation backed by GemFire. This store supports creating, updating,
 * deleting, and similarity searching of documents in a GemFire index.
 *
 * @author Anthony Baker
 */
public class GemFireVectorStore implements VectorStore {

	private static final Logger logger = LoggerFactory.getLogger(GemFireVectorStore.class);

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
			String base = UriComponentsBuilder
				.fromUriString("http{ssl}://{host}:{port}/gemfire-vectordb/v1/indexes/{index}")
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

	private static final int DEFAULT_PORT = 9090;

	private static final int DEFAULT_TOP_K_PER_BUCKET = 50;

	private static final String DEFAULT_DOCUMENT_FIELD = "document";

	public static final String INDEX_NAME = "spring-ai-index";

	public GemFireVectorStore(GemFireVectorStoreConfig config, EmbeddingClient embedding) {
		Assert.notNull(config, "GemFireVectorStoreConfig must not be null");
		Assert.notNull(embedding, "EmbeddingClient must not be null");

		this.client = config.client;
		this.embeddingClient = embedding;

		this.topKPerBucket = config.topKPerBucket;
		this.documentField = config.documentField;
	}

	private static final class CreateRequest {

		private String name;

		@JsonProperty("beam-width")
		private int beamWidth = 100;

		@JsonProperty("max-connections")
		private int maxConnections = 16;

		@JsonProperty("vector-similarity-function")
		private String vectorSimilarityFunction = "COSINE";

		private final String[] fields = new String[] { "vector" };

		private final int buckets = 0;

		public CreateRequest() {
		}

		public CreateRequest(String name) {
			this.name = name;
		}

		public String getName() {
			return name;
		}

		public void setName(String name) {
			this.name = name;
		}

		public int getBeamWidth() {
			return beamWidth;
		}

		public void setBeamWidth(int beamWidth) {
			this.beamWidth = beamWidth;
		}

		public int getMaxConnections() {
			return maxConnections;
		}

		public void setMaxConnections(int maxConnections) {
			this.maxConnections = maxConnections;
		}

		public String getVectorSimilarityFunction() {
			return vectorSimilarityFunction;
		}

		public void setVectorSimilarityFunction(String vectorSimilarityFunction) {
			this.vectorSimilarityFunction = vectorSimilarityFunction;
		}

	}

	private static final class UploadRequest {

		private final List<Embedding> embeddings;

		public List<Embedding> getEmbeddings() {
			return embeddings;
		}

		@JsonCreator
		public UploadRequest(@JsonProperty("embeddings") List<Embedding> embeddings) {
			this.embeddings = embeddings;
		}

		private static final class Embedding {

			private final String key;

			private List<Float> vector;

			@JsonInclude(JsonInclude.Include.NON_NULL)
			private Map<String, Object> metadata;

			public Embedding(@JsonProperty("key") String key, @JsonProperty("vector") List<Float> vector,
					String contentName, String content, @JsonProperty("metadata") Map<String, Object> metadata) {
				this.key = key;
				this.vector = vector;
				this.metadata = new HashMap<>(metadata);
				this.metadata.put(contentName, content);
			}

			public String getKey() {
				return key;
			}

			public List<Float> getVector() {
				return vector;
			}

			public Map<String, Object> getMetadata() {
				return metadata;
			}

		}

	}

	private static final class QueryRequest {

		@JsonProperty("vector")
		@NonNull
		private final List<Float> vector;

		@JsonProperty("top-k")
		private final int k;

		@JsonProperty("k-per-bucket")
		private final int kPerBucket;

		@JsonProperty("include-metadata")
		private final boolean includeMetadata;

		public QueryRequest(List<Float> vector, int k, int kPerBucket, boolean includeMetadata) {
			this.vector = vector;
			this.k = k;
			this.kPerBucket = kPerBucket;
			this.includeMetadata = includeMetadata;
		}

		public List<Float> getVector() {
			return vector;
		}

		public int getK() {
			return k;
		}

		public int getkPerBucket() {
			return kPerBucket;
		}

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
			List<Float> floatVector = document.getEmbedding().stream().map(Double::floatValue).toList();
			return new UploadRequest.Embedding(document.getId(), floatVector, documentField, document.getContent(),
					document.getMetadata());
		}).toList());

		ObjectMapper objectMapper = new ObjectMapper();
		String embeddingsJson = null;
		try {
			embeddingsJson = objectMapper.writeValueAsString(upload).substring("{\"embeddings\":".length());
		}
		catch (JsonProcessingException e) {
			throw new RuntimeException("Embedding json parsing error :" + e.getMessage());
		}

		client.post()
			.uri("/" + INDEX_NAME + "/embeddings")
			.contentType(MediaType.APPLICATION_JSON)
			.bodyValue(embeddingsJson)
			.retrieve()
			.bodyToMono(Void.class)
			.onErrorMap(WebClientException.class, this::handleHttpClientException)
			.block();
	}

	private Throwable handleHttpClientException(Throwable ex) {
		if (!(ex instanceof WebClientResponseException)) {
			logger.warn("Got an unexpected error: {}", ex.toString());
			return ex;
		}

		WebClientResponseException clientException = (WebClientResponseException) ex;

		if (clientException.getStatusCode().equals(NOT_FOUND)) {
			throw new RuntimeException("Index " + INDEX_NAME + " not found :" + ex);
		}
		else if (clientException.getStatusCode().equals(BAD_REQUEST)) {
			throw new RuntimeException("Bad Request :" + ex);
		}
		else {
			logger.warn("Got an unexpected HTTP error: {}, ", clientException.getStatusCode());
			return ex;
		}
	}

	@Override
	public Optional<Boolean> delete(List<String> idList) {
		try {
			client.method(HttpMethod.DELETE)
				.uri("/" + INDEX_NAME + "/embeddings/")
				.body(BodyInserters.fromValue(idList))
				.retrieve()
				.bodyToMono(Void.class)
				.block();
		}
		catch (Exception e) {
			logger.warn("Error removing embedding: " + e);
			return Optional.of(false);
		}
		return Optional.of(true);
	}

	@Override
	public List<Document> similaritySearch(SearchRequest request) {
		try {
			List<Double> vector = this.embeddingClient.embed(request.getQuery());
			List<Float> floatVector = vector.stream().map(Double::floatValue).toList();

			return client.post()
				.uri("/" + INDEX_NAME + "/query")
				.contentType(MediaType.APPLICATION_JSON)
				.bodyValue(new QueryRequest(floatVector, request.getTopK(), topKPerBucket, true))
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
		catch (Exception e) {
			throw new RuntimeException("Error in similarity search", e);
		}
	}

	public void createIndex() {
		try {
			CreateRequest createRequest = new CreateRequest();
			createRequest.setName(INDEX_NAME);
			createRequest.setBeamWidth(20);
			createRequest.setMaxConnections(16);
			ObjectMapper objectMapper = new ObjectMapper();
			String index = objectMapper.writeValueAsString(createRequest);
			client.post()
				.contentType(MediaType.APPLICATION_JSON)
				.bodyValue(index)
				.retrieve()
				.bodyToMono(Void.class)
				.block();
		}
		catch (Exception e) {
			logger.warn("An unexpected error occurred while creating the index");
		}
	}

	public void deleteIndex() {
		try {
			DeleteRequest deleteRequest = new DeleteRequest();
			deleteRequest.setDeleteData(true);
			client.method(HttpMethod.DELETE)
				.uri("/" + INDEX_NAME)
				.body(BodyInserters.fromValue(deleteRequest))
				.retrieve()
				.bodyToMono(Void.class)
				.block();
		}
		catch (Exception e) {
			logger.warn("An unexpected error occurred while deleting the index", e);
		}
	}

	private class DeleteRequest {

		@JsonProperty("delete-data")
		private boolean deleteData = true;

		public DeleteRequest() {
		}

		public DeleteRequest(boolean deleteData) {
			this.deleteData = deleteData;
		}

		public boolean isDeleteData() {
			return deleteData;
		}

		public void setDeleteData(boolean deleteData) {
			this.deleteData = deleteData;
		}

	}

}
