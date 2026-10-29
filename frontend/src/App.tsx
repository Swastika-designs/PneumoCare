import { Link, Route, Routes } from "react-router-dom";
import { EmptyState } from "./components/ui";
import { AppLayout } from "./layouts/AppLayout";
import AdvancedAnalytics from "./pages/AdvancedAnalytics";
import ClinicalText from "./pages/ClinicalText";
import Dashboard from "./pages/Dashboard";
import EhrExplorer from "./pages/EhrExplorer";
import ImageLab from "./pages/ImageLab";
import ModelPerformance from "./pages/ModelPerformance";
import Operations from "./pages/Operations";
import PneumoniaDetection from "./pages/PneumoniaDetection";
import Registry from "./pages/Registry";
import Reports from "./pages/Reports";
import { TutorialDetail, TutorialLibrary } from "./pages/TutorialLibrary";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="pneumonia" element={<PneumoniaDetection />} />
        <Route path="model-performance" element={<ModelPerformance />} />
        <Route path="image-lab" element={<ImageLab />} />
        <Route path="ehr" element={<EhrExplorer />} />
        <Route path="operations" element={<Operations />} />
        <Route path="clinical-text" element={<ClinicalText />} />
        <Route path="advanced" element={<AdvancedAnalytics />} />
        <Route path="tutorials" element={<TutorialLibrary />} />
        <Route path="tutorials/:id" element={<TutorialDetail />} />
        <Route path="registry" element={<Registry />} />
        <Route path="reports" element={<Reports />} />
        <Route path="*" element={<EmptyState title="Page not found" message="This page does not exist." action={<Link className="btn-primary" to="/">Go to overview</Link>} />} />
      </Route>
    </Routes>
  );
}
