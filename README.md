# React Practice

A code-reading practice project. Each lesson has clean, working React code.
Your job is to **understand** it: read the code, answer the questions,
then run the app and check your predictions.

See `REACT_TRACK.md` for the full topic list and progress.

## First-time setup (on your computer)

Needs Node.js 20.19+ or 22.12+.

```bash
git clone https://github.com/erenkemaldilmen/Customer-Purchase-Prediction-Using-Machine-Learning.git react-practice
cd react-practice
git checkout kemal/happy-rubin-kn3fql
npm install
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173).

## Each practice round

1. Get the new lesson from the cloud session:
   ```bash
   git pull
   ```
2. Open `src/lessons/<newest lesson>/` in your editor:
   - the `.jsx` file is the code to read
   - `QUESTIONS.md` has the questions
3. Answer in the chat, or write under "Your answers" in `QUESTIONS.md`
   and push (`git add -A && git commit -m "answers" && git push`).
4. In the browser, pick the lesson and click **Reveal output** to check
   your predictions. Predict first, then reveal.

## Layout

```
src/
  App.jsx                lesson picker + "Reveal output" scenarios
  lessons/
    index.js             list of lessons
    01-jsx-basics/
      ProductCard.jsx    the code to read
      demo.jsx           example inputs shown in the app
      QUESTIONS.md       questions + review notes
    ...
```
