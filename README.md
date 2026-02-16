# 🐷 Swine Feed Optimizer API

FastAPI service for optimizing swine feed formulations using linear programming.

## Files
- `main.py` - FastAPI application
- `requirements.txt` - Python dependencies
- `Procfile` - Railway start command
- `railway.toml` - Railway configuration

## Deploy to Railway via GitHub

### Step 1: Create GitHub Repository
1. Go to https://github.com/new
2. Create a new repository named "swine-optimizer"
3. Make it Public or Private (your choice)
4. Don't add README, .gitignore, or license (we have files already)

### Step 2: Upload Files to GitHub
You can use GitHub's web interface:
1. Go to your new repository
2. Click "uploading an existing file"
3. Upload all 5 files (main.py, requirements.txt, Procfile, railway.toml, README.md)
4. Commit directly to main

### Step 3: Deploy on Railway
1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Select your "swine-optimizer" repository
5. Railway will auto-detect Python and deploy!

## After Deployment

Railway will give you a URL like:
```
https://swine-optimizer-production.up.railway.app
```

### Test it:
```bash
curl https://YOUR-URL.railway.app/health
```

Should return: `{"status":"healthy"}`

### Update Supabase
In your PostgreSQL function, change:
```sql
v_python_url := 'https://YOUR-URL.railway.app/optimize';
```

## API Endpoints

- `GET /` - Service info
- `GET /health` - Health check  
- `POST /optimize` - Optimize feed formula

## No Environment Variables Needed! 🎉
