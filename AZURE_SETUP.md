# Azure OpenAI Setup Guide

This guide explains how to configure and use the Azure OpenAI integration in this project.

## Prerequisites

1. An Azure subscription
2. An Azure OpenAI resource deployed in your subscription
3. Deployed models for both embeddings and chat completions

## Azure OpenAI Setup

### 1. Create Azure OpenAI Resource

1. Go to the [Azure Portal](https://portal.azure.com/)
2. Create a new "Azure OpenAI" resource
3. Choose your subscription, resource group, and region
4. Select the pricing tier
5. Wait for deployment to complete

### 2. Deploy Models

You need to deploy two models:

1. **Embedding Model**: For text embeddings (e.g., `text-embedding-ada-002`)
2. **Chat Model**: For generating responses (e.g., `gpt-35-turbo` or `gpt-4`)

To deploy models:

1. Go to your Azure OpenAI resource
2. Navigate to "Model deployments"
3. Click "Create new deployment"
4. Select the model and give it a deployment name
5. Configure the deployment settings

### 3. Configure Environment Variables

Update the `.env` file with your Azure OpenAI details:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your-chat-deployment-name
```

### 4. Find Your Configuration Values

**Endpoint and API Key:**

1. Go to your Azure OpenAI resource in the Azure Portal
2. Navigate to "Keys and Endpoint"
3. Copy the endpoint URL and one of the API keys

**Deployment Names:**

1. Go to your Azure OpenAI resource
2. Navigate to "Model deployments"
3. Copy the deployment names you created

## Running the Code

Once configured, run the Azure OpenAI version:

```bash
uv run azure_test.py
```

## Common Model Deployments

- **Embeddings**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
- **Chat**: `gpt-35-turbo`, `gpt-4`, `gpt-4-turbo`

## Troubleshooting

1. **Authentication Error**: Check your API key and endpoint
2. **Model Not Found**: Verify your deployment names match the environment variables
3. **Rate Limiting**: Azure OpenAI has rate limits; reduce request frequency if needed
4. **Region Issues**: Some models may not be available in all regions

## Cost Considerations

- Embedding models are typically cheaper per token
- Chat models (especially GPT-4) can be more expensive
- Monitor usage in the Azure Portal under "Usage + quotas"
