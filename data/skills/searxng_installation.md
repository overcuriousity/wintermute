# SearXNG Installation and Usage

SearXNG is a privacy-respecting, open-source metasearch engine that aggregates results from multiple search engines without tracking users.

## Installation Summary

A working SearXNG instance has been installed at: `/home/user01/searxng-test/`

### Quick Start
```bash
cd ~/searxng-test
./start-searxng.sh
```

Then access at: http://127.0.0.1:8888

## Installation Details

### Dependencies Installed
- Python virtual environment with all required packages
- SearXNG cloned from GitHub

### Configuration
- Settings file: `~/searxng-test/searxng/etc/searxng/simple-settings.yml`
- Secret key: Generated and secured
- Running on: 127.0.0.1:8888
- Debug mode: Enabled (change to False for production)
- Valkey/Redis: Disabled (not required for basic operation)

### Key Files
1. `~/searxng-test/start-searxng.sh` - Startup script
2. `~/searxng-test/searxng.service` - Systemd service file
3. `~/searxng-test/searxng/etc/searxng/simple-settings.yml` - Configuration

## Usage

### Basic Usage
1. Start SearXNG: `./start-searxng.sh`
2. Open browser to: http://127.0.0.1:8888
3. Search as you would with any search engine

### Advanced Configuration

#### Enable More Search Engines
Edit `simple-settings.yml` and change `disabled: true` to `disabled: false` for desired engines.

#### Change Port
Modify `server.port` in `simple-settings.yml`

#### Make Accessible on Network
Change `server.bind_address` from `"127.0.0.1"` to `"0.0.0.0"` in `simple-settings.yml`

#### Set Up as System Service
```bash
sudo cp ~/searxng-test/searxng.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable searxng
sudo systemctl start searxng
```

## Notes
- The installation uses a Python virtual environment for isolation
- No Valkey/Redis is required for basic operation (caching disabled)
- For production use:
  - Disable debug mode
  - Set up proper secret key
  - Consider enabling Valkey for caching
  - Set up reverse proxy (nginx/apache)
  - Enable rate limiting