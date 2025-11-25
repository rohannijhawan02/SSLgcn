# 🚀 How to Start ToxPredict Web Application

## Quick Start Guide

### Method 1: Using Batch Files (EASIEST - RECOMMENDED)

1. **Start Backend:**
   - Double-click: `webapp\backend\start.bat`
   - A terminal window will open - **KEEP IT OPEN**
   - Wait until you see: "Application startup complete"

2. **Start Frontend:**
   - Open PowerShell in `webapp` folder
   - Run: `.\start-frontend.ps1`
   - OR manually: `cd webapp\frontend` then `npm run dev`

3. **Access Website:**
   - Open browser: http://localhost:3000
   - Backend API docs: http://localhost:8000/docs

---

### Method 2: Using PowerShell Scripts

Open **TWO separate PowerShell terminals**:

**Terminal 1 - Backend:**
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp
.\start-backend.ps1
```

**Terminal 2 - Frontend:**
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp
.\start-frontend.ps1
```

---

### Method 3: Manual Commands

**Backend (Terminal 1):**
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\backend
C:\Users\geeta\Anaconda3\envs\sslgcn_new\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Frontend (Terminal 2):**
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\frontend
npm run dev
```

---

## ⚠️ Important Notes

- **Keep both terminal windows OPEN** while using the application
- Backend must start BEFORE frontend
- If you see "network error", check that backend is running on port 8000
- To stop: Press `Ctrl+C` in each terminal

---

## 🔧 Troubleshooting

### "Port already in use" error

**Kill old processes:**
```powershell
# Find processes on port 8000 (backend)
netstat -ano | findstr ":8000"

# Find processes on port 3000 (frontend)
netstat -ano | findstr ":3000"

# Kill process (replace PID with actual process ID)
Stop-Process -Id <PID> -Force
```

### Backend won't start

1. Check conda environment exists:
   ```powershell
   conda env list
   ```
   Should show: `sslgcn_new`

2. Try running backend batch file directly:
   ```powershell
   cd webapp\backend
   .\start.bat
   ```

### Frontend won't start

1. Check node_modules exists:
   ```powershell
   cd webapp\frontend
   dir node_modules
   ```

2. If missing, reinstall:
   ```powershell
   npm install
   ```

---

## 📝 URLs

- **Frontend (Main Website):** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **API Interactive Docs:** http://localhost:8000/redoc

---

## ✅ Verification

To check if servers are running:

```powershell
# Check backend
netstat -ano | findstr "LISTENING" | findstr ":8000"

# Check frontend  
netstat -ano | findstr "LISTENING" | findstr ":3000"

# Test backend API
Invoke-WebRequest -Uri "http://localhost:8000/api/endpoints"
```

---

**Last Updated:** October 15, 2025
