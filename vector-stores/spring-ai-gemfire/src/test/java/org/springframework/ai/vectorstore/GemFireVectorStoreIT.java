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

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.embedding.TransformersEmbeddingClient;
import org.springframework.ai.vectorstore.GemFireVectorStore.GemFireVectorStoreConfig;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.DefaultResourceLoader;

/**
 * @author Anthony Baker
 */
// @EnabledIfEnvironmentVariable(named = "GEMFIRE_HOST", matches = ".+")
public class GemFireVectorStoreIT {

	List<Document> documents = List.of(
			new Document("1", getText("classpath:/test/data/spring.ai.txt"), Map.of("meta1", "meta1")),
			new Document("2", getText("classpath:/test/data/time.shelter.txt"), Map.of()),
			new Document("3", getText("classpath:/test/data/great.depression.txt"), Map.of("meta2", "meta2")));

	public static String getText(String uri) {
		var resource = new DefaultResourceLoader().getResource(uri);
		try {
			return resource.getContentAsString(StandardCharsets.UTF_8);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
		.withUserConfiguration(TestApplication.class);

	@BeforeEach
	public void createIndex() {
		contextRunner.run(c -> c.getBean(GemFireVectorStore.class).createIndex());
	}

	@AfterEach
	public void deleteIndex() {
		contextRunner.run(c -> c.getBean(GemFireVectorStore.class).deleteIndex());
	}

	@Test
	public void addAndSearchTest() {
		contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);
			vectorStore.add(documents);

			List<Document> results = vectorStore.similaritySearch(SearchRequest.query("Great Depression").withTopK(1));
			assertThat(results).hasSize(1);

			Document resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(documents.get(2).getId());
			assertThat(resultDoc.getContent()).contains("The Great Depression (1929–1939) was an economic shock");
			assertThat(resultDoc.getMetadata()).hasSize(2);
			assertThat(resultDoc.getMetadata()).containsKey("meta2");
			assertThat(resultDoc.getMetadata()).containsKey("distance");
		});
	}

	@Test
	public void documentUpdateTest() {
		contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			Document document = new Document(UUID.randomUUID().toString(), "Spring AI rocks!!",
					Collections.singletonMap("meta1", "meta1"));

			vectorStore.add(List.of(document));

			SearchRequest springSearchRequest = SearchRequest.query("Spring").withTopK(5);
			List<Document> results = vectorStore.similaritySearch(springSearchRequest);

			assertThat(results).hasSize(1);

			Document resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(document.getId());
			assertThat(resultDoc.getContent()).isEqualTo("Spring AI rocks!!");
			assertThat(resultDoc.getMetadata()).containsKey("meta1");
			assertThat(resultDoc.getMetadata()).containsKey("distance");

			Document sameIdDocument = new Document(document.getId(),
					"The World is Big and Salvation Lurks Around the Corner",
					Collections.singletonMap("meta2", "meta2"));

			vectorStore.add(List.of(sameIdDocument));
			SearchRequest fooBarSearchRequest = SearchRequest.query("FooBar").withTopK(5);
			results = vectorStore.similaritySearch(fooBarSearchRequest);

			assertThat(results).hasSize(1);

			resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(document.getId());
			assertThat(resultDoc.getContent()).isEqualTo("The World is Big and Salvation Lurks Around the Corner");
			assertThat(resultDoc.getMetadata()).containsKey("meta2");
			assertThat(resultDoc.getMetadata()).containsKey("distance");
		});
	}

	@Test
	public void searchThresholdTest() {

		contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);
			vectorStore.add(documents);

			List<Document> fullResult = vectorStore
				.similaritySearch(SearchRequest.query("Depression").withTopK(5).withSimilarityThresholdAll());

			List<Float> distances = fullResult.stream().map(doc -> (Float) doc.getMetadata().get("distance")).toList();
			assertThat(distances).hasSize(3);

			float threshold = (distances.get(0) + distances.get(1)) / 2;
			List<Document> results = vectorStore
				.similaritySearch(SearchRequest.query("Depression").withTopK(5).withSimilarityThreshold(1 - threshold));

			assertThat(results).hasSize(1);

			Document resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(documents.get(2).getId());
			assertThat(resultDoc.getContent()).contains("The Great Depression (1929–1939) was an economic shock");
			assertThat(resultDoc.getMetadata()).containsKey("meta2");
			assertThat(resultDoc.getMetadata()).containsKey("distance");
		});
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration
	public static class TestApplication {

		@Bean
		public GemFireVectorStoreConfig gemfireVectorStoreConfig() {
			// return
			// GemFireVectorStoreConfig.builder().withHost(System.getenv("GEMFIRE_HOST")).build();
			return GemFireVectorStoreConfig.builder().withHost("127.0.0.1").withIndex("spring-ai-index").build();
		}

		@Bean
		public GemFireVectorStore vectorStore(GemFireVectorStoreConfig config, EmbeddingClient embeddingClient) {
			return new GemFireVectorStore(config, embeddingClient);
		}

		@Bean
		public EmbeddingClient embeddingClient() {
			return new TransformersEmbeddingClient();
		}

	}

}