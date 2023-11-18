# Chat API
A flexible approach to integrating AI

## What
This is a API built to provide AI Agent based on company data sources and system prompts who also has access to custom defined tools.
The ingress and egress for the Agent is currently through a Slackbot and a REST API (FastAPI). 
Full [roadmap](https://github.com/salkinstitute/chat_api/edit/main/planning.md) still being determined.

## Why
Things are moving very fast with AI and it's potential keeps expanding.  This is a hands-on way to explore that potential and work directly with several different approaches with AI simultaneously until best practices are established.


### Features
- Slack Bot access
- Auto S3 and NoSQL (Mongo) retention / ingress of data
- Chat request NLP functions that include data ingress, note taking, note recall
- Mongo Express, FastAPI, Slackbot and Traefik edge routing all configured in this stack.
- Separate AI Agents for each business unit - allows for independent personas, expense tracking and tuning.
    
    
### Installation
- Stub for instructions on Docker and hosts file (w/Mkcert) setup

### Deployment
Not there yet, but getting close 
(will be using EC2)
Need to get auth based security setup first
