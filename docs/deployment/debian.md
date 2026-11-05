# Guia de Deploy Nativo em Produção — Debian Linux 12 (Bookworm)

**Projeto:** I.See Agents — Multi-Agent AI Platform  
**Autor:** Júlio César | **Organização:** I.See Technologies (`https://iseetechnologies.com.br/`)  
**Repositório:** `securityMaker488/isee-agents`  
**Escopo do Documento:** Manual completo de implantação nativa (*bare-metal* ou VM Linux) no **Debian 12 (Bookworm)**, cobrindo preparação de sistema operacional, PostgreSQL 16 + extensão `pgvector`, serviços `systemd` com isolamento de privilégios para o Backend FastAPI e Ollama, compilação do Frontend React/Next.js e configuração de proxy reverso Nginx com suporte a *Server-Sent Events (SSE)*.

---

## 1. Arquitetura de Produção no Debian 12

```mermaid
flowchart LR
    CLIENT["Navegador Web<br/>(HTTPS / TLS 1.3)"] --> NGINX["Nginx Reverse Proxy<br/>(:80 / :443)"]
    
    subgraph DEBIAN["Servidor Debian 12 (Bookworm)"]
        NGINX -->|"/ -> Static / SSR"| FE["Frontend Web<br/>(Build de Produção)"]
        NGINX -->|"/api/v1/* (Proxy + SSE)"| UVICORN["systemd: isee-backend.service<br/>(Gunicorn + Uvicorn Workers :8000)"]
        UVICORN <-->|"SQLAlchemy Async (:5432)"| PG["systemd: postgresql.service<br/>(PostgreSQL 16 + pgvector)"]
        UVICORN <-->|"HTTP Async (:11434)"| OLLAMA["systemd: ollama.service<br/>(Motor de Inferência Local)"]
        UVICORN -->|"Leitura/Escrita Isolada"| STORAGE["/var/lib/isee-agents/uploads<br/>(Permissão 0750 isee:isee)"]
    end
```

---

## 2. Preparação do Sistema Operacional e Usuário de Serviço

Execute os comandos abaixo para atualizar o sistema, instalar dependências de compilação e criar um usuário de sistema dedicado e sem shell de login (`isee`) para rodar a aplicação sob o princípio do menor privilégio:

```bash
# 1. Atualização de pacotes base e ferramentas de rede/compilação
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  curl wget git build-essential libpq-dev python3 python3-venv python3-dev \
  nginx ufw ca-certificates gnupg lsb-release

# 2. Criação do usuário e grupo dedicados da aplicação
sudo groupadd --system isee
sudo useradd --system --gid isee --create-home --home-dir /opt/isee-agents --shell /usr/sbin/nologin isee

# 3. Criação do diretório persistente de uploads e logs com permissão restrita (0750)
sudo mkdir -p /var/lib/isee-agents/uploads /var/log/isee-agents
sudo chown -R isee:isee /var/lib/isee-agents /var/log/isee-agents
sudo chmod -R 750 /var/lib/isee-agents /var/log/isee-agents
```

---

## 3. Instalação e Configuração do PostgreSQL + `pgvector`

No Debian 12 (Bookworm), instalamos o PostgreSQL e a extensão `pgvector` a partir dos repositórios oficiais do Debian ou PGDG:

```bash
# 1. Instalar PostgreSQL e extensão pgvector
sudo apt install -y postgresql postgresql-contrib postgresql-15-pgvector || \
sudo apt install -y postgresql postgresql-contrib postgresql-16-pgvector

# 2. Garantir que o serviço esteja ativo e habilitado no boot
sudo systemctl enable --now postgresql

# 3. Criar banco de dados, usuário dedicado e habilitar a extensão vector
sudo -u postgres psql << 'EOF'
CREATE USER isee_user WITH ENCRYPTED PASSWORD 'ALTERE_PARA_SENHA_FORTE_AQUI';
CREATE DATABASE isee_agents OWNER isee_user;
\c isee_agents
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL PRIVILEGES ON DATABASE isee_agents TO isee_user;
GRANT ALL ON SCHEMA public TO isee_user;
EOF
```

Verifique se a extensão vetorial foi instalada corretamente:

```bash
sudo -u postgres psql -d isee_agents -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

---

## 4. Instalação e Hardening do Serviço Ollama (`systemd`)

```bash
# 1. Instalar o Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Configurar override do systemd para controle de memória e bind estrito em localhost
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null << 'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_ORIGINS=http://localhost:*,https://iseetechnologies.com.br"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_KEEP_ALIVE=15m"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF

# 3. Recarregar systemd e reiniciar o Ollama
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl restart ollama

# 4. Baixar os modelos base utilizados pelos agentes e pelo RAG
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

---

## 5. Implantação do Backend FastAPI e Variáveis de Ambiente

```bash
# 1. Clonar o repositório em /opt/isee-agents
sudo -u isee git clone https://github.com/securityMaker488/isee-agents.git /opt/isee-agents/app
cd /opt/isee-agents/app/backend

# 2. Criar ambiente virtual Python e instalar dependências de produção
sudo -u isee python3 -m venv /opt/isee-agents/app/backend/.venv
sudo -u isee /opt/isee-agents/app/backend/.venv/bin/pip install --upgrade pip
sudo -u isee /opt/isee-agents/app/backend/.venv/bin/pip install -r requirements.txt
```

### 5.1 Arquivo de Variáveis de Ambiente (`/etc/isee-agents/backend.env`)

Crie o arquivo de segredos fora da árvore pública e proteja-o com permissão `0600`:

```bash
sudo mkdir -p /etc/isee-agents
sudo tee /etc/isee-agents/backend.env > /dev/null << 'EOF'
APP_NAME="I.See Agents"
ENVIRONMENT="production"
DEBUG="false"

# Banco de Dados PostgreSQL + pgvector
DATABASE_URL="postgresql+asyncpg://isee_user:ALTERE_PARA_SENHA_FORTE_AQUI@127.0.0.1:5432/isee_agents"

# Segurança JWT (gere com: openssl rand -hex 32)
SECRET_KEY="GERE_UMA_CHAVE_HEXADECIMAL_DE_64_CARACTERES_COM_OPENSSL_RAND"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES="60"

# Integração Ollama Local
OLLAMA_BASE_URL="http://127.0.0.1:11434"
DEFAULT_LLM_MODEL="qwen2.5:7b"
DEFAULT_EMBEDDING_MODEL="nomic-embed-text"

# Diretório de Uploads e Limites
UPLOAD_DIR="/var/lib/isee-agents/uploads"
MAX_UPLOAD_SIZE_MB="20"
CORS_ORIGINS='["https://agentes.iseetechnologies.com.br"]'
EOF

sudo chown isee:isee /etc/isee-agents/backend.env
sudo chmod 600 /etc/isee-agents/backend.env
```

### 5.2 Execução das Migrações de Banco (Alembic)

```bash
cd /opt/isee-agents/app/backend
sudo -u isee env $(cat /etc/isee-agents/backend.env | grep -v '^#' | xargs) \
  /opt/isee-agents/app/backend/.venv/bin/alembic upgrade head
```

### 5.3 Unidade `systemd` para o Backend FastAPI (`/etc/systemd/system/isee-backend.service`)

```ini
[Unit]
Description=I.See Agents FastAPI Backend Service
After=network.target postgresql.service ollama.service
Requires=postgresql.service

[Service]
Type=simple
User=isee
Group=isee
WorkingDirectory=/opt/isee-agents/app/backend
EnvironmentFile=/etc/isee-agents/backend.env
ExecStart=/opt/isee-agents/app/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4 --proxy-headers
Restart=always
RestartSec=5

# Diretivas de Hardening systemd
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/isee-agents/uploads /var/log/isee-agents

[Install]
WantedBy=multi-user.target
```

Ative o serviço:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now isee-backend.service
sudo systemctl status isee-backend.service
```

---

## 6. Build do Frontend e Configuração do Nginx Reverse Proxy

### 6.1 Compilação do Frontend

```bash
# Instalar Node.js 20 LTS (NodeSource)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

cd /opt/isee-agents/app/frontend
sudo -u isee npm ci
sudo -u isee VITE_API_BASE_URL="/api/v1" npm run build
```

### 6.2 Configuração do Nginx (`/etc/nginx/sites-available/isee-agents`)

O detalhe mais importante ao configurar o Nginx para o **I.See Agents** é desabilitar o `proxy_buffering` nas rotas de chat para permitir que o **streaming de tokens em tempo real (SSE)** chegue imediatamente ao navegador, além de ajustar `client_max_body_size` e `proxy_read_timeout` para chamadas de inferência:

```nginx
server {
    listen 80;
    server_name agentes.iseetechnologies.com.br;

    # Cabeçalhos de Segurança HTTP
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self';" always;

    # Limite de upload para documentos RAG e imagens multimodais
    client_max_body_size 25M;

    # Servir o Frontend compilado (SPA / Static Assets)
    root /opt/isee-agents/app/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy Reverso para o Backend FastAPI (/api/v1/*)
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Essencial para Streaming de Respostas do LLM (SSE) e Inferência Local
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

Ative o site e configure o firewall UFW:

```bash
sudo ln -sf /etc/nginx/sites-available/isee-agents /etc/nginx/sites-enabled/isee-agents
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# Firewall UFW: liberar apenas SSH, HTTP e HTTPS (nunca expor 5432, 8000 ou 11434 externamente)
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```
