# Chat AI
A flexible approach to integrating AI

## Scope & Deliverables

- JWT Auth API to allow data ingest & remote integration of chat, with RBAC
- Slackbot with NL commands for most API endpoints
- Safe keeping of paid-for stuff (embeddings, ai responses) and trainable data (chat history, votes) 
- Free(ish) RAG via Pinecone on lexical datatypes (pdf, word, doc, txt)
- NL analysis on reporting data sources (csv, xls, json, powerbi) with simple tables (markdown?) or charts (seaborn)
- Mongo for it's own datastore - chat history, prompts, personas, user notes, user roles, bot tools (later in db, in py for now) - on backup schedule
- Data source integration with Box, GitHub, S3 and local MySQL (fed through Xplenty?)
- CICD through either GitHub Actions or GitLab
- Abstracted, layer-up approach to interacting with LLMs as the model and type is a moving target. 
- AutoGen autonomous agent 'Documentarian'

### Components
**Maybe refactor to use the official FastAPI cookiecutter, then add my AI/ML secret sauce:  https://github.com/tiangolo/full-stack-fastapi-postgresql**

#### API
The API will be written in the lang. that gives the most options and support for interacting with the LLMs - this is currently python.
FastAPI will be used for it's perf. , wide-support, self-documenting and ease of use.  The API will contain all code and logic for the every interface (API, slack, remote inte.).  LangChain Agents will be used to do the ML heavy lifting with extensively used custom tools (functions that we write) so that they can be reused and role-based. 

#### Slackbot
Again, python for options and support of ML and the official slack_bolt python package.

#### Data
- Local app : Mongo 
- Local Reporting : Mysql and Mongo
- File storage: s3
- File ingest: Box, Slack attachments, s3 (allows easy Lambda for event based ingest)

#### TODO for CICD with Github Actions
[] Docker build and push workflow : https://docs.github.com/en/actions/publishing-packages/publishing-docker-images
[] Docker-compose for ec2 *use overrides?*
[] Deploy workflow, maybe: easingthemes/ssh-deploy@main






