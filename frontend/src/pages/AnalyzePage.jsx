import PageWrapper from "../components/Layout/PageWrapper";

export default function AnalyzePage() {
  return (
    <PageWrapper>
      <h1 className="text-2xl font-bold text-textPrimary mb-2">Upload &amp; Configure</h1>
      <p className="text-textSecondary">
        Upload your CSV dataset and configure target and sensitive attributes.
        This page will be expanded in the next build step.
      </p>
    </PageWrapper>
  );
}
