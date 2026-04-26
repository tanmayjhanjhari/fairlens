import PageWrapper from "../components/Layout/PageWrapper";

export default function ResultsPage() {
  return (
    <PageWrapper>
      <h1 className="text-2xl font-bold text-textPrimary mb-2">Analysis Results</h1>
      <p className="text-textSecondary">
        Fairness metrics, audit score, and visual dashboard will appear here.
        This page will be expanded in the next build step.
      </p>
    </PageWrapper>
  );
}
