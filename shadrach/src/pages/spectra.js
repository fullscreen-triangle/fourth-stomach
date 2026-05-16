import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import StockDashboard from "@/components/charts/StockDashboard";

const SpectralMatchingTool = dynamic(
  () => import("@/components/tools/SpectralMatchingTool"),
  { ssr: false, loading: () => <div className="h-96 rounded-2xl bg-surface/30 animate-pulse" /> }
);

export default function SpectraPage() {
  return (
    <>
      <Head>
        <title>Universal Spectral Matching | Fourth Stomach</title>
        <meta name="description" content="Compare any two portfolios as spectral images. WebGL2 fragment shader interference — no database required." />
      </Head>
      <TransitionEffect />
      <main className="w-full min-h-screen">
        <Layout>
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <p className="font-mono text-xs tracking-widest text-primary/60 uppercase mb-3">
              Portfolio Method Comparison — Interactive Tool
            </p>
            <h1 className="text-4xl font-bold text-light mb-3 lg:text-3xl md:text-2xl">
              Universal Spectral Matching
            </h1>
            <p className="text-light/50 max-w-2xl mb-10 leading-relaxed">
              Every bounded persistent system is oscillatory (Oscillatory Necessity Theorem).
              Portfolios are converted to spectra{" "}
              <span className="font-mono text-primary/70">{"{"}&omega;k, Ak, &phi;k{"}"}</span>,
              then to 2D spectral images: frequency → x, phase → y, amplitude → brightness.
              A WebGL2 fragment shader computes pixel-wise interference between any two images — no
              database, no lookup. Comparison IS the computation.
            </p>
            <SpectralMatchingTool />
            <StockDashboard
              name="SPECTRAL SIMILARITY INDEX"
              primaryColor="#d4a843"
              negColor="#e8734a"
              seed={137}
            />
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
