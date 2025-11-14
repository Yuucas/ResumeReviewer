# Troubleshooting Guide

## Issue: "timeout of 300000ms exceeded" and 500 Internal Server Error

### Symptoms
- Search request times out after 5 minutes
- Backend returns 500 error
- Console shows: "timeout of 300000ms exceeded"

### Root Causes

#### 1. Ollama is not running or models are not loaded
**Check:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return list of available models
```

**Fix:**
```bash
# Start Ollama (if not running)
ollama serve

# Pull required models (in separate terminal)
ollama pull nomic-embed-text
ollama pull qwen3:latest
```

#### 2. LLM Processing is too slow
The system analyzes each candidate with an LLM, which can be slow.

**Quick Fix - Reduce number of candidates:**
In your search, reduce `top_k` from 5 to 2:
```javascript
// In SearchForm or search parameters
top_k: 2  // Instead of 5
```

#### 3. Backend timeout is too short
**Fix:** Already set to 300 seconds (5 minutes) in `frontend/src/services/api.js`

### Immediate Solutions

#### Option 1: Check Ollama Status
1. Open terminal
2. Run: `ollama list`
3. Verify both models are installed:
   - `nomic-embed-text`
   - `qwen3:latest`
4. If not, run:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3:latest
   ```

#### Option 2: Use Smaller Search Parameters
1. Search for only 2 candidates instead of 5+
2. Add specific filters (role_category, min_experience)

#### Option 3: Check Backend Logs
Look at the terminal running the backend for the actual error:
```
2025-11-13 XX:XX:XX - ERROR - Search failed: <actual error here>
```

### Common Errors and Fixes

#### Error: "Connection refused" to Ollama
```bash
# Start Ollama
ollama serve
```

#### Error: "Model not found"
```bash
# Pull the missing model
ollama pull qwen3:latest
```

#### Error: "Out of memory"
- Close other applications
- Reduce `top_k` to 2
- Use a smaller LLM model

### Performance Tips

1. **First Search is Slow** - Ollama loads models on first use (1-2 min)
2. **Subsequent Searches** - Should be faster (30s - 2min depending on top_k)
3. **Reduce Candidates** - Searching for 2 candidates is 2.5x faster than 5

### Debug Mode

To see detailed logs, check the backend terminal output. It shows:
- Which step is running
- How many candidates found
- Processing time for each stage

### Still Having Issues?

1. Check backend terminal for specific error messages
2. Verify Ollama is responding: `curl http://localhost:11434/api/tags`
3. Try with top_k=1 to test if it works with minimal load
4. Check system resources (RAM, CPU)
