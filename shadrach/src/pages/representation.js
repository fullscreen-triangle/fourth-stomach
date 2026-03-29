import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

// Dynamic imports (D3 needs browser APIs)
const CircuitGraph = dynamic(() => import("@/components/charts/CircuitGraph"), { ssr: false })
const GasMolecules = dynamic(() => import("@/components/charts/GasMolecules"), { ssr: false })
const SequenceStream = dynamic(() => import("@/components/charts/SequenceStream"), { ssr: false })
const TransformCycle = dynamic(() => import("@/components/charts/TransformCycle"), { ssr: false })

const Section = ({ id, title, children }) => (
  <motion.section
    id={id}
    className="mb-16"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
  >
    <h2 className="text-2xl font-bold text-primary mb-4 border-b border-primary/20 pb-2">{title}</h2>
    <div className="text-light/70 leading-relaxed space-y-4">{children}</div>
  </motion.section>
);

const Theorem = ({ name, statement }) => (
  <div className="border-l-4 border-primary/50 pl-4 my-4 bg-surface/50 py-3 pr-4 rounded-r-lg">
    <p className="text-primary font-semibold text-sm mb-1">{name}</p>
    <p className="text-light/80 text-sm italic">{statement}</p>
  </div>
);

const ChartPanel = ({ title, children }) => (
  <motion.div
    className="border border-primary/20 rounded-xl bg-surface/30 p-4 overflow-hidden"
    initial={{ opacity: 0, scale: 0.95 }}
    whileInView={{ opacity: 1, scale: 1 }}
    viewport={{ once: true }}
  >
    <p className="text-xs font-mono text-primary/60 uppercase tracking-widest mb-3">{title}</p>
    <div className="flex justify-center">{children}</div>
  </motion.div>
);

export default function Representation() {
  return (
    <>
      <Head>
        <title>Multi-Modal Representation | Fourth Stomach</title>
        <meta name="description" content="Financial System Representation Through Multi-Modal Transformation: Circuit Networks, Sequence Encoding, and Gas Molecular Dynamics" />
      </Head>
      <TransitionEffect />
      <main className="w-full mb-16 flex flex-col items-center justify-center text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Multi-Modal Financial Representation"
            className="!text-4xl mb-4 xl:!text-3xl lg:!text-2xl"
          />
          <p className="text-center text-light/50 mb-12 max-w-3xl mx-auto">
            The same financial network admits three mathematically equivalent representations &mdash;
            circuit, sequence, and gas molecular &mdash; connected by invertible transformations
            that preserve &gt;95% of information across full round-trip cycles.
          </p>

          <div className="max-w-5xl mx-auto">

            {/* Interactive Visualizations */}
            <div className="grid grid-cols-2 gap-6 mb-16 lg:grid-cols-1">
              <ChartPanel title="Circuit Representation">
                <CircuitGraph width={460} height={350} />
              </ChartPanel>
              <ChartPanel title="Gas Molecular Dynamics">
                <GasMolecules width={460} height={350} nMolecules={25} />
              </ChartPanel>
              <ChartPanel title="Sequence Encoding">
                <SequenceStream width={460} height={350} />
              </ChartPanel>
              <ChartPanel title="Transformation Cycle">
                <TransformCycle width={460} height={350} />
              </ChartPanel>
            </div>

            <Section id="circuit" title="1. Circuit Network Representation">
              <p>
                The financial transaction network maps to an electrical circuit
                C = (N, E, I, V) where nodes are entities, edges are transaction channels,
                I : E &rarr; R assigns currents (transaction flow rates), and
                V : N &rarr; R assigns potentials (economic values).
              </p>
              <Theorem
                name="Kirchhoff's Current Law (Capital Conservation)"
                statement="At any node i at time t: the sum of all incoming transaction flows equals the sum of all outgoing flows. Inflow = Outflow in the conservation regime."
              />
              <Theorem
                name="Kirchhoff's Voltage Law (No-Arbitrage)"
                statement="For any closed loop in the transaction network, the sum of potential differences around the loop is zero. Violation would imply risk-free profit from circular trading."
              />
              <p>
                Three circuit elements model distinct financial instruments:
                <strong> Resistors</strong> (R) represent transaction friction (bid-ask spread, fees);
                <strong> Capacitors</strong> (C) represent liquidity buffers (reserves, margin);
                <strong> Inductors</strong> (L) represent trading momentum (order flow persistence).
              </p>
              <Theorem
                name="Shadow Edges as Quantum Coherence"
                statement="Virtual transactions in the shadow network correspond to quantum-coherent circuit elements with S-values > 1.0. When |ρ_ij| > 0.8, constructive interference yields superposition states where the shadow current exceeds any individual component."
              />
            </Section>

            <Section id="sequence" title="2. Sequence Representation">
              <p>
                Transactions are encoded as directional sequences where each transaction
                maps to one of six cardinal directions based on its properties:
                <strong> North</strong> (large amount),
                <strong> South</strong> (small amount),
                <strong> East</strong> (frequent pattern),
                <strong> West</strong> (rare pattern),
                <strong> Up</strong> (high profit),
                <strong> Down</strong> (low profit).
              </p>
              <p>
                The sequence representation enables <strong>semantic distance amplification</strong>
                through four encoding layers, each multiplying the distance between similar
                and dissimilar patterns by a layer-specific factor:
              </p>
              <div className="grid grid-cols-4 gap-3 my-4 md:grid-cols-2">
                {[
                  { name: 'Directional', gamma: '3.7x', color: 'text-primary' },
                  { name: 'Positional', gamma: '4.2x', color: 'text-coral' },
                  { name: 'Frequency', gamma: '5.8x', color: 'text-gold' },
                  { name: 'Compressed', gamma: '7.3x', color: 'text-navy' },
                ].map((layer, i) => (
                  <div key={i} className="p-3 bg-surface border border-primary/10 rounded-lg text-center">
                    <p className="text-light/40 text-xs">{layer.name}</p>
                    <p className={`${layer.color} font-bold text-lg`}>{layer.gamma}</p>
                  </div>
                ))}
              </div>
              <p>
                Total amplification: &Gamma; = &prod; &gamma;<sub>i</sub> &asymp; <strong>658x</strong>.
                This enables LLM-style continuous learning where the system predicts
                future transactions from directional sequence context.
              </p>
            </Section>

            <Section id="gas" title="3. Gas Molecular Representation">
              <p>
                The transaction network maps to a gas molecular system G = (M, &Psi;, H)
                where molecules are nodes, wavefunctions &Psi; encode transaction patterns,
                and the Hamiltonian H represents total system energy. Each molecule&apos;s position
                is its location in S-entropy space (S<sub>knowledge</sub>, S<sub>time</sub>, S<sub>entropy</sub>).
              </p>
              <Theorem
                name="Harmonic Coincidence"
                statement="Molecules i and j interfere when |n&omega;_i - m&omega;_j| < &epsilon;_tol for integers n,m, creating correlation &rho;_ij = |<&Psi;_i|&Psi;_j>|. This is the inner product of their wavefunctions in Hilbert space."
              />
              <Theorem
                name="Financial Equilibrium as Maxwell-Boltzmann"
                statement="At equilibrium, the node value distribution follows P(V_i) = Z^{-1} exp(-&beta;V_i), where &beta; = 1/(k_B T_market) is inverse market temperature and Z is the partition function."
              />
              <p>
                The chamber geometry is defined by the metric tensor g<sub>&mu;&nu;</sub>
                whose components are the pairwise correlations &rho;<sub>ij</sub>. This geometry
                determines wave propagation, interference patterns, and the approach to equilibrium.
              </p>
            </Section>

            <Section id="equivalence" title="4. Representational Equivalence">
              <p>
                Three transformation operators connect the modalities:
              </p>
              <div className="bg-surface border border-primary/20 rounded-lg p-4 my-4 font-mono text-sm text-center space-y-1">
                <p className="text-primary">T<sub>C&rarr;S</sub> : Circuit &rarr; Sequence</p>
                <p className="text-coral">T<sub>S&rarr;G</sub> : Sequence &rarr; Gas</p>
                <p className="text-gold">T<sub>G&rarr;C</sub> : Gas &rarr; Circuit</p>
              </div>
              <Theorem
                name="Composition Identity"
                statement="T_{G&rarr;C} &circ; T_{S&rarr;G} &circ; T_{C&rarr;S} = T_equiv, where T_equiv is identity up to representational gauge. The round-trip transformation preserves all information."
              />
              <Theorem
                name="Information Conservation Across Modalities"
                statement="I(T) = I(C) = I(S) = I(G). Information content remains invariant under representation transformations because each transformation is invertible (bijective)."
              />
              <p>
                These are not metaphors but <strong>mathematically equivalent frameworks</strong>.
                Circuit analysis reveals conservation and flow bottlenecks. Sequence analysis
                enables pattern prediction and anomaly detection. Gas analysis reveals
                thermodynamic equilibrium and correlation structure. Information discovered
                in one modality transfers to the others through the proven isomorphisms.
              </p>
            </Section>

            <Section id="applications" title="5. Applications">
              <div className="grid grid-cols-3 gap-4 md:grid-cols-1">
                <div className="p-4 bg-surface border border-primary/10 rounded-lg">
                  <h4 className="text-primary font-bold mb-2">Real-Time Analysis</h4>
                  <p className="text-sm text-light/50">
                    Circuit: bottlenecks and liquidity pools.
                    Sequence: predict next transaction patterns.
                    Gas: observe equilibrium approach, identify hot/cold regions.
                  </p>
                </div>
                <div className="p-4 bg-surface border border-coral/10 rounded-lg">
                  <h4 className="text-coral font-bold mb-2">Risk Identification</h4>
                  <p className="text-sm text-light/50">
                    Circuit: cascading failure modes.
                    Sequence: anomalous pattern similarity.
                    Gas: tightly-coupled molecular clusters.
                  </p>
                </div>
                <div className="p-4 bg-surface border border-gold/10 rounded-lg">
                  <h4 className="text-gold font-bold mb-2">Optimal Intervention</h4>
                  <p className="text-sm text-light/50">
                    Circuit: where to inject liquidity.
                    Sequence: predict intervention effects.
                    Gas: equilibrium-restoring rebalancing.
                  </p>
                </div>
              </div>
            </Section>

          </div>
        </Layout>
      </main>
    </>
  );
}
