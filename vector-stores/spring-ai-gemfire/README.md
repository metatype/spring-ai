# GemFire Vector Store

This readme walks you through setting up the GemFire `VectorStore` to store document embeddings and perform similarity searches.

## What is GemFire?

[GemFire](https://tanzu.vmware.com/gemfire) is an ultra high speed in-memory data and compute grid, with vector extensions to store and search vectors efficiently.

## Prerequisites

1. OpenAI Account: Create an account at [OpenAI Signup](https://platform.openai.com/signup) and generate the token at [API Keys](https://platform.openai.com/account/api-keys).

2. Access to a GemFire cluster with the GemFire Vector Database extension installed

// TODO

## Configuration

// TODO create an index

When setting up embeddings, select a vector dimension of `1536`. This matches the dimensionality of OpenAI's model `text-embedding-ada-002`, which we'll be using for this guide.

Additionally, you'll need to provide your OpenAI API Key. Set it as an environment variable like so:

```bash
export SPRING_AI_OPENAI_API_KEY='Your_OpenAI_API_Key'
```

## Repository

To acquire Spring AI artifacts, declare the Spring Snapshot repository:

```xml
<repository>
	<id>spring-snapshots</id>
	<name>Spring Snapshots</name>
	<url>https://repo.spring.io/snapshot</url>
	<releases>
		<enabled>false</enabled>
	</releases>
</repository>
```

## Dependencies

Add these dependencies to your project:

1. OpenAI: Required for calculating embeddings.

```xml
<dependency>
	<groupId>org.springframework.experimental.ai</groupId>
	<artifactId>spring-ai-openai-spring-boot-starter</artifactId>
	<version>0.7.0-SNAPSHOT</version>
</dependency>
```

2. GemFire

```xml
<dependency>
    <groupId>org.springframework.experimental.ai</groupId>
    <artifactId>spring-ai-gemfire</artifactId>
    <version>0.7.0-SNAPSHOT</version>
</dependency>
```

## Sample Code

To configure GemFire in your application, you can use the following setup:

```java
@Bean
public GemFireVectorStoreConfig gemFireVectorStoreConfig() {
    return GemFireVectorStoreConfig.builder()
        .withUrl("http://localhost:8080")
        .withIndexName("spring-ai-test-index")
        .build();
}
```

Integrate with OpenAI's embeddings by adding the Spring Boot OpenAI starter to your project.
This provides you with an implementation of the Embeddings client:

```java
@Bean
public VectorStore vectorStore(GemFireVectorStoreConfig config, EmbeddingClient embeddingClient) {
    return new GemFireVectorStore(config, embeddingClient);
}
```

In your main code, create some documents:

```java
List<Document> documents = List.of(
	new Document("Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!!", Map.of("meta1", "meta1")),
	new Document("The World is Big and Salvation Lurks Around the Corner"),
	new Document("You walk forward facing the past and you turn back toward the future.", Map.of("meta2", "meta2")));
```

Add the documents to Pinecone:

```java
vectorStore.add(List.of(document));
```

And finally, retrieve documents similar to a query:

```java
List<Document> results = vectorStore.similaritySearch("Spring", 5);
```

If all goes well, you should retrieve the document containing the text "Spring AI rocks!!".
