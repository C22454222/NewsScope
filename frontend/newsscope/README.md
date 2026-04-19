# 📱 NewsScope Frontend

NewsScope is the **Flutter frontend** for a cross‑platform news aggregation and bias‑analysis system. It provides a clean, responsive interface for Android and Web, connecting to the FastAPI backend and Supabase database. The app allows users to sign up, log in, and explore global news stories with bias detection, sentiment analysis, and fact‑checking integration.

---

## 🚀 Features

- **Firebase Authentication**  
  - Email/Password and Google OAuth login/signup  
  - Secure session management with JWT tokens  

- **News Aggregation UI**  
  - Displays stories ingested from NewsAPI, GDELT, and RSS feeds  
  - Interactive comparison of outlets across ideological spectrum  

- **Bias & Sentiment Visualisation**  
  - Articles positioned on left–right scale  
  - Sentiment indicators (positive/negative/neutral)  

- **Fact‑Checking Integration**  
  - Inline summaries from PolitiFact and other APIs  
  - Highlighted claims with credibility scores  

- **User Bias Profile**  
  - Personalised dashboard showing reading habits  
  - Blind‑spot suggestions for under‑represented perspectives  

---

## 🛠️ Tech Stack

- **Framework:** Flutter (Dart)  
- **State Management:** Provider / Riverpod (planned)  
- **Backend:** FastAPI (Python) hosted on Render  
- **Database:** Supabase Postgres + Buckets  
- **Auth:** Firebase Authentication (Email/Google OAuth)  
- **CI/CD:** GitHub Actions (planned)  

---

## 📂 Project Structure

```structure
frontend/newsscope/
├── lib/
│   ├── main.dart          # Entry point, Firebase init
│   ├── auth_gate.dart     # Switches between login and home
│   ├── sign_in_screen.dart# Login/signup UI
│   ├── home_screen.dart   # Post-login dashboard
│   └── ...                # Future UI modules
├── android/               # Android platform code
├── web/                   # Web platform code
├── pubspec.yaml           # Dependencies
```

---

## ⚙️ Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/newsscope.git
   cd newsscope/frontend/newsscope
   ```

2. **Install dependencies**#

   ```bash
   flutter pub get
   ```

3. **Configure Firebase**
   - Create a Firebase project in [Firebase Console](https://console.firebase.google.com/).
   - Enable Email/Password and Google sign‑in.
   - Download `google-services.json` and place it in `android/app/`.

4. **Run the app**

   ```bash
   flutter run
   ```

---

## 🧪 Testing

- Unit tests for UI widgets and auth flows in `/test`.
- Integration tests planned for API calls and bias visualisation.

---

## 📌 Roadmap

- [ ] Connect frontend to FastAPI endpoints (articles, fact‑checks).  
- [ ] Implement bias spectrum visualisation widget.  
- [ ] Add personalised bias profile dashboard.  
- [ ] Deploy companion web version.  

---

## 📖 License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.
