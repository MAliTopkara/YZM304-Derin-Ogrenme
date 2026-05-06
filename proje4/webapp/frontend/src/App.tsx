import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import ModelSelectPage from "./pages/ModelSelectPage";
import ResultsPage from "./pages/ResultsPage";
import UploadPage from "./pages/UploadPage";

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/model" element={<ModelSelectPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}

export default App;
