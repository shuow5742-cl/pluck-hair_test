import { Route, Routes, Navigate } from "react-router-dom";
import HomePage from "@/pages/HomePage";
import SettingsPage from "@/pages/SettingsPage";
import TestConsolePage from "@/pages/TestConsolePage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<TestConsolePage />} />
      <Route path="/legacy" element={<HomePage />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
