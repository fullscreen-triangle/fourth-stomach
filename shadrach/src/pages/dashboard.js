import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const Dashboard = dynamic(
  () => import("@/components/Dashboard"),
  { ssr: false, loading: () => <div className="h-96 rounded-2xl bg-surface/30 animate-pulse" /> }
);

export default function DashboardPage() {
  return (
    <>
      <Head>
        <title>Transaction Clock Dashboard | Fourth Stomach</title>
        <meta name="description" content="Live market data through the transaction clock framework. Monetary derivatives, spectral interference, and S-entropy scoring — all in-browser via WebGL2." />
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
              Full-Stack Market Intelligence — Live or Synthetic Data
            </p>
            <h1 className="text-4xl font-bold text-light mb-3 lg:text-3xl md:text-2xl">
              Transaction Clock Dashboard
            </h1>
            <p className="text-light/50 max-w-2xl mb-4 leading-relaxed">
              Every price series is transformed through the{" "}
              <span className="font-mono text-primary/70">transaction clock</span>{" "}
              Θ(t) = ∫|G(s)|ds, replacing calendar time with accumulated gain-loss
              intensity. The monetary derivative dP/dΘ normalises velocity by
              activity — quiet periods are amplified, volatile ones compressed.
              The spectral interference panel compares the real series against a
              theoretical subordinated Brownian motion at the same volatility:
              constructive patterns (teal) show where model and market align;
              destructive (coral) show where the framework sees structure the
              conventional view misses.
            </p>
            <p className="text-light/30 text-xs font-mono mb-10">
              Enter an Alpha Vantage API key (free at alphavantage.co) for live data — or leave blank to use synthetic data.
            </p>
            <Dashboard />
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
