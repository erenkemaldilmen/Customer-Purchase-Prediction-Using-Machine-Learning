# 4. Conditional rendering — `Inbox.jsx`

Status: in progress

Read `Inbox.jsx`, answer below (or in the chat), then run the app and reveal
the scenarios to check yourself.

## Questions
1. **Early returns:** if `isLoading` is `true` **and** `error` also exists, what shows?
   Why do you think the order is loading → error → user?
2. **Scenario A:** `user = { name: "Kemal" }`, not loading, no error, 3 messages:
   `"Invoice"` (read), `"Meeting"` (unread), `"Hello"` (read). Describe the screen.
3. **Scenario B:** same user, `messages = []`. What shows? Does the notice appear?
4. **`unreadCount > 0 &&`:** why `> 0` instead of just `{unreadCount && (...)}`?
   Hint: what does React show for the **number** `0`?
5. **`&&` vs ternary:** why `&&` for the notice but `? :` for the list?

## Your answers
1.
2.
3.
4.
5.
