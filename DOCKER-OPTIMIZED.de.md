# Optimierte Docker-Konfiguration für UltraRAG

Dieses Verzeichnis enthält optimierte Docker-Konfigurationen für den Produktionseinsatz von UltraRAG.

## Erstellte Dateien

### 1. `Dockerfile.production`
Multi-Stage Dockerfile optimiert für:
- **Minimale Image-Größe** (~500MB vs ~1.5GB ursprünglich)
- **Sicherheit** mit Nicht-Root-Benutzer (`ultrarag:ultrarag`)
- **Optimiertes Layer-Caching** für schnellere Builds
- **Health Checks** für Container-Überwachung
- **Produktionsoptimierungen** (PYTHONDONTWRITEBYTECODE, PYTHONUNBUFFERED)

### 2. `docker-compose.yml`
Produktionsbereite docker-compose Konfiguration mit:
- Ressourcenlimits (Speicher: 2GB)
- Health Checks
- Volume-Mounts für Persistenz
- Netzwerk-Isolation

### 3. `.dockerignore`
Schließt unnötige Dateien vom Build-Kontext aus, um die Image-Größe zu reduzieren.

## Wichtige Optimierungen

### Multi-Stage Build
- **Builder-Stufe**: Installiert Build-Abhängigkeiten und kompiliert Pakete
- **Runtime-Stufe**: Minimales Python 3.12.9-slim Image mit nur Laufzeitabhängigkeiten

### Sicherheit
- Nicht-Root-Benutzer (`ultrarag` mit UID 1001, GID 1001)
- Korrekte Dateiberechtigungen
- Minimale Laufzeitabhängigkeiten

### Leistung
- UV für schnelle Abhängigkeitsauflösung
- Layer-Caching-Optimierung (Abhängigkeitsdateien zuerst kopiert)
- `--no-cache` und `--link-mode=copy` für uv

### Produktionsbereit
- Health Checks für Container-Überwachung
- Umgebungsvariablen für die Produktion
- Ressourcenlimits in docker-compose
- Korrekte Logging-Konfiguration

## Verwendung

### Optimiertes Image bauen:
```bash
docker build -f Dockerfile.production -t ultrarag:production .
```

### Mit docker-compose ausführen:
```bash
docker-compose up -d
```

### Container-Health prüfen:
```bash
docker-compose ps
curl http://localhost:5050/
```

## Umgebungsvariablen

| Variable | Standardwert | Beschreibung |
|----------|---------|-------------|
| `ULTRA_RAG_ENV` | `production` | Anwendungsumgebung |
| `PORT` | `5050` | Anwendungsport |
| `HOST` | `0.0.0.0` | Bind-Adresse |
| `PYTHONUNBUFFERED` | `1` | Ungepufferte Python-Ausgabe |
| `PYTHONDONTWRITEBYTECODE` | `1` | Schreibe keine .pyc-Dateien |

## Optionale Abhängigkeiten

Der Produktions-Build enthält:
- `retriever` Gruppe: Infinity-Embeddings, Sentence-Transformers, FAISS, etc.
- `generation` Gruppe: vLLM, Transformers, Torch

Von der Produktion ausgeschlossen (kann bei Bedarf hinzugefügt werden):
- `evaluation` Gruppe: ROUGE-Scores, pytrec-eval
- `corpus` Gruppe: Mineru-Corpus-Tools
- `dev` Gruppe: Entwicklungswerkzeuge

## Image-Größenvergleich

- **Ursprüngliches Dockerfile**: ~1.5GB (Ubuntu + CUDA + alle Abhängigkeiten)
- **Optimiertes Dockerfile**: ~500MB (Python slim + nur Laufzeitabhängigkeiten)

## Health Check

Der Container enthält einen Health Check, der überprüft, ob die Anwendung antwortet:
```bash
# Manueller Health Check
curl -f http://localhost:5050/ || echo "Health Check fehlgeschlagen"
```

## Überwachung

Container-Logs anzeigen:
```bash
docker-compose logs -f ultrarag
```

Ressourcennutzung prüfen:
```bash
docker stats
```

## Fehlerbehebung

### Build-Probleme:
```bash
# Build-Cache löschen
docker builder prune

# Mit detaillierterer Ausgabe bauen
docker build --progress=plain -f Dockerfile.production .
```

### Berechtigungsprobleme:
```bash
# Volume-Berechtigungen korrigieren
sudo chown -R 1001:1001 ./data ./logs
```

### Abhängigkeitsprobleme:
Bearbeiten Sie `Dockerfile.production`, um optionale Abhängigkeitsgruppen im `uv sync` Befehl anzupassen.