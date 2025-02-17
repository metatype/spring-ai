package org.springframework.ai.vectorstore;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

import com.theokanning.openai.client.OpenAiApi;
import com.theokanning.openai.service.OpenAiService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.testcontainers.containers.Neo4jContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory;
import retrofit2.converter.jackson.JacksonConverterFactory;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.openai.embedding.OpenAiEmbeddingClient;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Gerrit Meier
 * @author Michael Simons
 * @author Christian Tzolov
 */
@Testcontainers
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
class Neo4jVectorStoreIT {

	// Neo4j 5.12 has a bug wrt checking limits, so either 5.11 or anything higher than
	// 5.12 works
	@Container
	static Neo4jContainer<?> neo4jContainer = new Neo4jContainer<>(DockerImageName.parse("neo4j:5.11"))
		.withRandomPassword();

	List<Document> documents = List.of(
			new Document("Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!!",
					Collections.singletonMap("meta1", "meta1")),
			new Document("Hello World Hello World Hello World Hello World Hello World Hello World Hello World"),
			new Document(
					"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression",
					Collections.singletonMap("meta2", "meta2")));

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
		.withUserConfiguration(TestApplication.class);

	@BeforeEach
	void cleanDatabase() {
		this.contextRunner
			.run(context -> context.getBean(Driver.class).executableQuery("MATCH (n) DETACH DELETE n").execute());
	}

	@Test
	void addAndSearchTest() {
		this.contextRunner.run(context -> {

			VectorStore vectorStore = context.getBean(VectorStore.class);

			vectorStore.add(this.documents);

			List<Document> results = vectorStore.similaritySearch(SearchRequest.query("Great").withTopK(1));

			assertThat(results).hasSize(1);
			Document resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
			assertThat(resultDoc.getContent()).isEqualTo(
					"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
			assertThat(resultDoc.getMetadata()).containsKey("meta2");
			assertThat(resultDoc.getMetadata()).containsKey("distance");

			// Remove all documents from the store
			vectorStore.delete(this.documents.stream().map(Document::getId).toList());

			List<Document> results2 = vectorStore.similaritySearch(SearchRequest.query("Great").withTopK(1));
			assertThat(results2).isEmpty();
		});
	}

	@Test
	void documentUpdateTest() {

		this.contextRunner.run(context -> {

			VectorStore vectorStore = context.getBean(VectorStore.class);

			Document document = new Document(UUID.randomUUID().toString(), "Spring AI rocks!!",
					Collections.singletonMap("meta1", "meta1"));

			vectorStore.add(List.of(document));

			List<Document> results = vectorStore.similaritySearch(SearchRequest.query("Spring").withTopK(5));

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

			results = vectorStore.similaritySearch(SearchRequest.query("FooBar").withTopK(5));

			assertThat(results).hasSize(1);
			resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(document.getId());
			assertThat(resultDoc.getContent()).isEqualTo("The World is Big and Salvation Lurks Around the Corner");
			assertThat(resultDoc.getMetadata()).containsKey("meta2");
			assertThat(resultDoc.getMetadata()).containsKey("distance");

		});
	}

	@Test
	void searchThresholdTest() {

		this.contextRunner.run(context -> {

			VectorStore vectorStore = context.getBean(VectorStore.class);

			vectorStore.add(this.documents);

			List<Document> fullResult = vectorStore
				.similaritySearch(SearchRequest.query("Great").withTopK(5).withSimilarityThresholdAll());

			List<Float> distances = fullResult.stream().map(doc -> (Float) doc.getMetadata().get("distance")).toList();

			assertThat(distances).hasSize(3);

			float threshold = (distances.get(0) + distances.get(1)) / 2;

			List<Document> results = vectorStore
				.similaritySearch(SearchRequest.query("Great").withTopK(5).withSimilarityThreshold(1 - threshold));

			assertThat(results).hasSize(1);
			Document resultDoc = results.get(0);
			assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
			assertThat(resultDoc.getContent()).isEqualTo(
					"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
			assertThat(resultDoc.getMetadata()).containsKey("meta2");
			assertThat(resultDoc.getMetadata()).containsKey("distance");

		});
	}

	@Test
	void ensureVectorIndexGetsCreated() {
		this.contextRunner.run(context -> {
			assertThat(context.getBean(Driver.class)
				.executableQuery(
						"SHOW indexes yield name, type WHERE name = 'spring-ai-document-index' AND type = 'VECTOR' return count(*) > 0")
				.execute()
				.records()
				.get(0) // get first record
				.get(0)
				.asBoolean()) // get returned result
				.isTrue();
		});
	}

	@Test
	void ensureIdIndexGetsCreated() {
		this.contextRunner.run(context -> {
			assertThat(context.getBean(Driver.class)
				.executableQuery(
						"SHOW indexes yield labelsOrTypes, properties, type WHERE any(x in labelsOrTypes where x = 'Document')  AND any(x in properties where x = 'id') AND type = 'RANGE' return count(*) > 0")
				.execute()
				.records()
				.get(0) // get first record
				.get(0)
				.asBoolean()) // get returned result
				.isTrue();
		});
	}

	@SpringBootConfiguration
	@EnableAutoConfiguration(exclude = { DataSourceAutoConfiguration.class })
	public static class TestApplication {

		@Bean
		public VectorStore vectorStore(Driver driver, EmbeddingClient embeddingClient) {

			return new Neo4jVectorStore(driver, embeddingClient,
					Neo4jVectorStore.Neo4jVectorStoreConfig.defaultConfig());
		}

		@Bean
		public Driver driver() {
			return GraphDatabase.driver(neo4jContainer.getBoltUrl(),
					AuthTokens.basic("neo4j", neo4jContainer.getAdminPassword()));
		}

		@Bean
		public EmbeddingClient embeddingClient() {

			Retrofit retrofit = new Retrofit.Builder().baseUrl("https://api.openai.com")
				.client(OpenAiService.defaultClient(System.getenv("OPENAI_API_KEY"), Duration.ofSeconds(60)))
				.addConverterFactory(JacksonConverterFactory.create(OpenAiService.defaultObjectMapper()))
				.addCallAdapterFactory(RxJava2CallAdapterFactory.create())
				.build();

			OpenAiApi api = retrofit.create(OpenAiApi.class);

			return new OpenAiEmbeddingClient(new OpenAiService(api), "text-embedding-ada-002");
		}

	}

}
