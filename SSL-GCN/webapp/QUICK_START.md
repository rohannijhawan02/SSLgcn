# 🎯 ToxPredict - Quick Reference Card

## 🚀 FASTEST WAY TO START (Double-Click)

```
📁 webapp\START_ALL.bat    ← Just double-click this!
```

This will:
- ✅ Start backend server (port 8000)
- ✅ Start frontend server (port 3000)
- ✅ Open your browser automatically

---

## 🔧 Alternative: Start Individually

### Option 1: Batch Files (Easiest)

1. **Backend:** Double-click `webapp\backend\start.bat`
2. **Frontend:** Run `webapp\start-frontend.ps1` in PowerShell

### Option 2: PowerShell Scripts

Open PowerShell in `webapp` folder:

```powershell
# Terminal 1 - Backend
.\start-backend.ps1

# Terminal 2 - Frontend (open NEW terminal)
.\start-frontend.ps1
```

---

## 🌐 Access URLs

| Service | URL |
|---------|-----|
| **Website** | http://localhost:3000 |
| **API** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |

---

## 🛑 How to STOP

- Press `Ctrl+C` in each terminal window
- OR close the terminal windows

---

## ❗ If Something Goes Wrong

### Network Error on Website
```powershell
# Check if backend is running
netstat -ano | findstr ":8000"
```
**Fix:** Start the backend server

### Port Already in Use
```powershell
# Kill processes on port 8000
Stop-Process -Id <PID> -Force
```
**Fix:** Find PID using netstat command above

### Need More Help?
📖 Read: `webapp\HOW_TO_START.md`

---

## ✅ Quick Health Check

```powershell
# Are servers running?
netstat -ano | findstr ":8000"  # Backend
netstat -ano | findstr ":3000"  # Frontend

# Test backend
Invoke-WebRequest http://localhost:8000/api/endpoints
```

---

**Remember:** Both servers must be running for the website to work!
