import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import StockDashboard from "@/components/charts/StockDashboard";

const MarketTrajectoryViz = dynamic(
  () => import("@/components/tools/MarketTrajectoryViz"),
  { ssr: false, loading: () => <div className="h-96 rounded-2xl bg-surface/30 animate-pulse" /> }
);

export default function TrajectoryPage() {
  return (
    <>
      <Head>
        <title>Market Phase-Space Trajectory | Fourth Stomach</title>
        <meta name="description" content="Watch market trajectories close in bounded phase space. Poincaré recurrence visualised in real time." />
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
              Market Dynamics — Interactive Tool
            </p>
            <h1 className="text-4xl font-bold text-light mb-3 lg:text-3xl md:text-2xl">
              Market Phase-Space Trajectory
            </h1>
            <p className="text-light/50 max-w-2xl mb-10 leading-relaxed">
              Bounded phase space implies Poincaré recurrence: every trajectory is a closed loop.
              Each asset is an Ornstein-Uhlenbeck oscillator — a damped harmonic oscillator with noise.
              The phase portrait (price <span className="font-mono text-primary/70">x</span> vs.
              velocity <span className="font-mono text-primary/70">v</span>) shows the trajectory
              winding through a stationary Gaussian distribution. Computation = traversal of this loop.
            </p>
            <MarketTrajectoryViz />
            <StockDashboard
              name="TRAJECTORY MARKET DATA"
              primaryColor="#2ca89a"
              negColor="#e8734a"
              seed={256}
            />
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
