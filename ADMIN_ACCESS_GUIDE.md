# Accessing Admin Lab - Troubleshooting Guide

## ✓ Step-by-Step Guide to Access Admin Lab

### Step 1: Login as Researcher
1. Go to http://localhost:5173
2. Click **"Launch Diagnostic Tool"**
3. **Select "Researcher"** (not Clinician)
4. You should see the button change to highlight "Researcher"
5. Keep credentials as default:
   - Username: `admin_lab`
   - Password: `password`
6. Click **"Access System"**

### Step 2: View Admin Lab Button
After successful login:
- Look at the **top navigation bar**
- You should see two buttons:
  - ✓ "Diagnostics" (always visible for all roles)
  - ✓ "Admin Lab" (ONLY for Researcher role)

### Step 3: Click Admin Lab
- Click the **"Admin Lab"** button in the navigation
- You should now see the Training Lab interface

---

## 🔍 If You Don't See "Admin Lab" Button

### Issue #1: Not Logged in as Researcher
**Solution:**
1. Click your username in top-right
2. Click "Logout"
3. Try login again
4. Make sure to click **"Researcher"** button before logging in
5. Verify it's highlighted in white/blue before submitting

### Issue #2: Browser Cache
**Solution:**
1. Hard refresh page: `Ctrl+F5` (Windows) or `Cmd+Shift+R` (Mac)
2. Or clear browser cache:
   - Chrome: Ctrl+Shift+Delete
   - Firefox: Ctrl+Shift+Delete
   - Safari: Command+Shift+Delete
3. Try logging in again

### Issue #3: Page Not Loading
**Solution:**
1. Check if dev server is running:
   - Look at terminal, should see: `  ➜  Local:   http://localhost:5173/`
2. If not running, run: `npm run dev`
3. Refresh browser: `F5`

### Issue #4: Network Issue
**Solution:**
1. Open browser console: `F12`
2. Go to Network tab
3. Refresh page
4. Look for any red errors
5. Try in different browser (Chrome, Firefox, Edge)

---

## 🧪 Testing Admin Lab Access

### Manual Test
```javascript
// Open browser console (F12)
// Paste this to check role:
localStorage.getItem('userRole')
// Should return: null (use URL state instead)
```

### URL-Based Navigation
If buttons aren't working, try accessing directly:
1. Login as Researcher first
2. Manually type in URL: `http://localhost:5173`
3. The app should remember you're logged in

---

## 📱 Complete Login Flow

```
Home Page
  ↓ Click "Launch Diagnostic Tool"
Login Page
  ↓ Select "Researcher"
  ↓ Keep default credentials
  ↓ Click "Access System"
Authenticated as Researcher
  ↓ See "Admin Lab" in navbar
  ↓ Click "Admin Lab"
Admin Training Lab Page
  ✓ Model configuration visible
  ✓ Training controls available
```

---

## 💻 What Admin Lab Should Look Like

When successfully logged in as Researcher, Admin Lab shows:

### Left Panel (Configuration)
- Dataset & Preprocessing section
- Network Architecture section
  - Model options (DenseNet169, ResNet50, etc.)
  - Optimizer selection
  - Learning rate control
  - Batch size
  - Epochs
- "Start Training" button

### Right Panel (Visualization)
- Progress bar (when training)
- Convergence metrics chart (blank until training starts)
- Training console output (blank until training starts)

---

## 🆘 Still Having Issues?

### Try These Debugging Steps:

1. **Clear Everything:**
   - Close browser tab
   - Close entire browser
   - Restart computer
   - Open http://localhost:5173 in new browser

2. **Restart Dev Server:**
   - Press `Ctrl+C` in terminal
   - Run: `npm run dev`
   - Wait 5 seconds
   - Refresh browser

3. **Check for Errors:**
   - Open browser console: `F12`
   - Go to Console tab
   - Look for red error messages
   - Screenshot the error
   - Check below for error solutions

4. **Try Different Browser:**
   - Chrome
   - Firefox
   - Edge
   - Safari

---

## 🐛 Common Errors & Solutions

### Error: "Cannot read property 'admin'"
**Cause:** Role not being passed correctly
**Solution:** Make sure you clicked "Researcher" button before login

### Error: "AdminDashboard is not defined"
**Cause:** Component not imported
**Solution:** Restart dev server (`npm run dev`)

### Error: "Unexpected token" in console
**Cause:** JavaScript syntax error
**Solution:** 
1. Check if linting passes: `npm run lint`
2. Restart dev server: `npm run dev`

### Blank Page After Login
**Cause:** Component rendering issue
**Solution:**
1. Hard refresh: `Ctrl+F5`
2. Clear cache: `Ctrl+Shift+Delete`
3. Restart browser

---

## ✅ Verify It's Working

After accessing Admin Lab successfully, test:

1. **Model Selection:**
   - Click dropdown under "Backbone Architecture"
   - Should see options: DenseNet169, ResNet50, MobileNetV2, EfficientNetB3

2. **Dataset Selection:**
   - Click dropdown under "Dataset"
   - Should see options: HAM10000, ISIC 2019, Custom Clinical Dataset

3. **Class Balancing Toggle:**
   - Click the toggle switch
   - Should slide left/right

4. **Start Training Button:**
   - Should be clickable (not grayed out)
   - When clicked, should trigger training

---

## 📞 Quick Checklist

- [ ] Visited http://localhost:5173
- [ ] Clicked "Launch Diagnostic Tool"
- [ ] Selected "Researcher" (button highlighted)
- [ ] Logged in with credentials
- [ ] See "Admin Lab" in top navigation
- [ ] Clicked "Admin Lab" button
- [ ] Admin Lab page loaded successfully
- [ ] Can interact with model selection dropdown
- [ ] Can interact with dataset dropdown
- [ ] Can interact with training button

---

## 🎯 Next Steps

Once you access Admin Lab:

1. **Try Model Training:**
   - Select DenseNet169
   - Click "Start Training"
   - Watch real-time metrics

2. **Explore Features:**
   - Try different architectures
   - Change hyperparameters
   - Watch convergence graphs

3. **Read Documentation:**
   - See QUICK_START.md for full guide
   - See BACKEND_INTEGRATION_GUIDE.md for technical details

---

**Still Stuck?** 
- Reload this page and follow steps 1-3 carefully
- Try in a different browser
- Restart your computer
- Check browser console for errors (F12)

