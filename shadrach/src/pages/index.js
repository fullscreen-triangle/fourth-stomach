import AnimatedText from "@/components/AnimatedText";
import { HireMe } from "@/components/HireMe";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const FeatureCard = ({ number, label, description }) => (
  <motion.div
    className="flex flex-col items-center p-6 border border-primary/20 rounded-xl bg-surface hover:border-primary/50 transition-colors"
    whileHover={{ y: -5 }}
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
  >
    <span className="text-5xl font-bold text-primary mb-2">{number}</span>
    <span className="text-xl font-semibold text-light mb-2">{label}</span>
    <p className="text-light/60 text-center text-sm">{description}</p>
  </motion.div>
);

const SystemCard = ({ title, tag, description }) => (
  <motion.div
    className="p-6 border border-primary/10 rounded-xl bg-surface/50 hover:border-primary/30 transition-all"
    whileHover={{ y: -3, boxShadow: "0 0 20px rgba(44,168,154,0.1)" }}
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
  >
    <span className="text-xs font-mono text-primary/70 uppercase tracking-widest">{tag}</span>
    <h3 className="text-lg font-bold text-light mt-1 mb-2">{title}</h3>
    <p className="text-light/50 text-sm leading-relaxed">{description}</p>
  </motion.div>
);

export default function Home() {
  return (
    <>
      <Head>
        <title>Fourth Stomach | Unified Economic Coordination Framework</title>
        <meta name="description" content="Markets are gases. Portfolios are circuits. One axiom, zero free parameters, fifteen validated predictions." />
      </Head>

      <TransitionEffect />

      <main className="flex items-center text-light w-full min-h-screen">
        <Layout className="pt-0 md:pt-16 sm:pt-8">
          {/* Hero */}
          <div className="flex flex-col items-center w-full mb-20">
            <motion.p
              className="text-primary font-mono text-sm tracking-widest uppercase mb-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              Derived from a single axiom
            </motion.p>

            <AnimatedText
              text="Markets Are Gases. Portfolios Are Circuits."
              className="!text-5xl xl:!text-4xl lg:!text-3xl md:!text-2xl text-center"
            />

            <motion.p
              className="text-light/60 text-lg max-w-3xl text-center mt-6 md:text-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
            >
              A computational framework grounded in bounded phase space, from which the complete
              thermodynamic, information-theoretic, and circuit-theoretic description of financial
              markets follows by mathematical necessity. One axiom. Zero adjustable parameters.
              Every prediction validated.
            </motion.p>

            <motion.div
              className="flex items-center gap-4 mt-8 md:flex-col"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.0 }}
            >
              <Link
                href="/portfolio"
                className="px-8 py-3 bg-primary text-dark font-semibold rounded-lg hover:bg-primaryDark transition-colors"
              >
                Read the Papers
              </Link>
              <Link
                href="/api-docs"
                className="px-8 py-3 border border-primary text-primary font-semibold rounded-lg hover:bg-primary/10 transition-colors"
              >
                Explore the API
              </Link>
              <Link
                href="/licensing"
                className="px-8 py-3 border border-light/20 text-light/60 font-semibold rounded-lg hover:border-primary/50 hover:text-primary transition-colors"
              >
                License
              </Link>
            </motion.div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-4 gap-6 mb-20 lg:grid-cols-2 md:grid-cols-1">
            <FeatureCard number="1" label="Axiom" description="Bounded phase space. Everything else follows." />
            <FeatureCard number="0" label="Free Parameters" description="No curve fitting. No adjustable constants. Pure mathematics." />
            <FeatureCard number="15/15" label="Predictions Validated" description="Every testable prediction confirmed experimentally." />
            <FeatureCard number="7" label="Core Systems" description="CTN, STN, GCF, Representations, Portfolio, DTI, Temporal Arbitrage." />
          </div>

          {/* The Implication Chain */}
          <div className="mb-20">
            <h2 className="text-3xl font-bold text-center mb-2">The Chain of Implications</h2>
            <p className="text-light/40 text-center mb-10 max-w-2xl mx-auto">
              From a single empirical fact, the entire framework follows by mathematical necessity.
            </p>
            <div className="flex flex-col items-center gap-2">
              {[
                "All financial quantities are finite (empirical fact)",
                "Bounded phase space (Axiom)",
                "Poincare recurrence (theorem)",
                "Oscillatory dynamics (corollary)",
                "Triple equivalence: Oscillation = Category = Partition",
                "Market-gas isomorphism (symplectic bijection)",
                "Complete statistical mechanics of markets",
                "Fuzzy circuit portfolio optimisation with time-invariant fixed point",
              ].map((step, i) => (
                <motion.div
                  key={i}
                  className="flex items-center gap-3 w-full max-w-2xl"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.08 }}
                >
                  <span className="text-primary font-mono text-sm w-6 shrink-0">{i + 1}.</span>
                  <div className="flex-1 py-3 px-4 bg-surface border border-primary/10 rounded-lg text-light/70 text-sm">
                    {step}
                  </div>
                  {i < 7 && <span className="text-primary/40 text-lg ml-2">&#8595;</span>}
                </motion.div>
              ))}
            </div>
          </div>

          {/* Seven Systems */}
          <div className="mb-20">
            <h2 className="text-3xl font-bold text-center mb-2">Seven Core Systems</h2>
            <p className="text-light/40 text-center mb-10">Unified by the fourth-stomach architecture: circulatory processing with bidirectional flow.</p>
            <div className="grid grid-cols-3 gap-4 lg:grid-cols-2 md:grid-cols-1">
              <SystemCard tag="Chamber I" title="Circulation Transaction Networks"
                description="Batch settlement via graph reduction. Kirchhoff's current law enforces conservation. O(n log n) complexity." />
              <SystemCard tag="Chamber II" title="Shadow Transaction Networks"
                description="FFT harmonic coincidence detection transforms transaction trees into correlation graphs revealing hidden market structure." />
              <SystemCard tag="Chamber III" title="Graph Completion Finance"
                description="Shadow network reveals missing flows. Directed loans complete the graph. The loan IS the flow." />
              <SystemCard tag="Representation" title="Multi-Modal Transformations"
                description="Circuit, sequence, gas molecular, and shadow representations. >95% information preservation across round-trips." />
              <SystemCard tag="Portfolio" title="Fuzzy Oscillatory Circuit Graphs"
                description="Contraction mapping on fuzzy state tuples converges to time-invariant optimal allocation via Banach fixed-point theorem." />
              <SystemCard tag="Index" title="Distributed Thermodynamic Index"
                description="The partition function of the market gas. Every observable is a derivative. Phase transitions predict crashes." />
              <SystemCard tag="Chamber IV" title="Temporal Arbitrage"
                description="Intraday capital optimisation. Settlement certainty >99%. Capital reuse multiplier ~3-4x per trading day." />
            </div>
          </div>

        </Layout>
      </main>

      <HireMe />
    </>
  );
}
