import Head from "next/head";
import Layout from "@/components/Layout";
import TransitionEffect from "@/components/TransitionEffect";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const GasOscillatorViz = dynamic(
  () => import("@/components/tools/GasOscillatorViz"),
  { ssr: false, loading: () => <div className="h-96 rounded-2xl bg-surface/30 animate-pulse" /> }
);

export default function OscillatorPage() {
  return (
    <>
      <Head>
        <title>Gas Oscillator Field | Fourth Stomach</title>
        <meta name="description" content="Interactive gas oscillator field in S-entropy space. Every market participant is an oscillator." />
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
              Thermodynamic Index — Interactive Tool
            </p>
            <h1 className="text-4xl font-bold text-light mb-3 lg:text-3xl md:text-2xl">
              Gas Oscillator Field
            </h1>
            <p className="text-light/50 max-w-2xl mb-10 leading-relaxed">
              Every market participant is an oscillator in S-entropy space{" "}
              <span className="font-mono text-primary/70">[0,1]³</span> — kinetic entropy Sk encodes
              temperature, temporal entropy St encodes complexity, energetic entropy Se encodes
              computational density. Kuramoto coupling drives synchronisation: when the field locks,
              markets move together.
            </p>
            <GasOscillatorViz />
          </motion.div>
        </Layout>
      </main>
    </>
  );
}
