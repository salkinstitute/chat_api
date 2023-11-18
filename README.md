# Chat API
A flexible approach to integrating AI

## What
This is a API built to provide AI Agent based on company data (Using RAG with a Pinecone vectorestore) sources and system prompts who also has access to custom defined tools.
The ingress and egress for the Agent is currently through a Slackbot and a REST API (FastAPI). 
Full [roadmap](https://github.com/salkinstitute/chat_api/edit/main/planning.md) still being determined.

## Why
Things are moving very fast with AI and it's potential keeps expanding.  This is a hands-on way to explore that potential and work directly with several different approaches with AI simultaneously until best practices are established.
Alot of the solutions out there for RAG are good but missing key things (metadata filters, hyperparameter access, etc).  Even more leaky abstractions dealing with LLMs currently, at this point we need hands-on control for some functions and the ability to re-use what's settled in others, -it's a highly fluid environment.


### Features
- Slack Bot access
- Auto S3 and NoSQL (Mongo) retention / ingress of data
- Chat request NLP functions that include data ingress, note taking, note recall
- Mongo Express, FastAPI, Slackbot and Traefik edge routing all configured in this stack.
- Separate AI Agents for each business unit - allows for independent personas, expense tracking and tuning.
    
    
### Installation
1. Install [Mkcert](https://github.com/FiloSottile/mkcert) and drop the certs in the certs/dev folder (replace what's there) update the names of the certs in the traefik section of the docker-compose file if the certs have a different name.
2. Add host entries
    
```
    #for Mac and *nix
    sudo vi /etc/hosts
    # Add a line at the end of your hosts file with all services in the stack:
    127.0.0.1    chat-api.local, traefik.chat-api.local, mex.chat-api.local

```

4. Clone `git clone https://github.com/salkinstitute/chat_api.git && cd chat_api`
5. Create your .env file from the example and update the values `cp example.env .env`
6. Make sure Docker is installed and running then `docker-compose up`
   
### Deployment
Not there yet, but getting close 
(will be using EC2)

