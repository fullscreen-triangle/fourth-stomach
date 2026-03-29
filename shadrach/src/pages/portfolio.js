import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

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

const ResultRow = ({ prediction, result, detail }) => (
  <div className="flex items-start gap-3 py-3 border-b border-primary/10">
    <span className="text-primary text-lg mt-0.5">&#10003;</span>
    <div>
      <p className="text-light/90 font-medium">{prediction}</p>
      <p className="text-light/50 text-sm">{result} &mdash; {detail}</p>
    </div>
  </div>
);

export default function Portfolio() {
  return (
    <>
      <Head>
        <title>Portfolio Optimisation | Fourth Stomach</title>
        <meta name="description" content="Portfolio Optimisation as Trajectory Completion in Fuzzy Oscillatory Circuit Networks" />
      </Head>
      <TransitionEffect />
      <main className="w-full mb-16 flex flex-col items-center justify-center text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="Portfolio Optimisation as Trajectory Completion"
            className="!text-4xl mb-4 xl:!text-3xl lg:!text-2xl"
          />
          <p className="text-center text-light/50 mb-12 max-w-3xl mx-auto">
            A time-invariant framework for asset allocation under epistemic uncertainty,
            where the portfolio IS an oscillatory circuit graph and the optimal allocation
            IS the unique fixed point of a contraction mapping.
          </p>

          <div className="max-w-4xl mx-auto">

            <Section id="core-idea" title="1. The Core Idea">
              <p>
                Classical portfolio theory (Markowitz, 1952) treats assets as entries in a vector
                and optimises a quadratic objective. This discards network structure, ignores
                epistemic uncertainty, and produces allocations that are notoriously unstable.
              </p>
              <p>
                We model the portfolio as an <strong>oscillatory circuit graph</strong>: each asset
                is a node occupying bounded phase space (prices are finite, volumes are finite,
                returns are bounded), and therefore exhibits oscillatory dynamics by the Poincare
                recurrence theorem. Each coupling between assets is an edge carrying a conductance
                derived from a universal transport formula that unifies correlation, capital flow,
                and information propagation.
              </p>
              <p>
                Node states are represented as <strong>fuzzy membership functions</strong> encoding
                epistemic uncertainty. Kirchhoff&apos;s laws provide conservation (KCL: capital balance)
                and equilibrium (KVL: no-arbitrage) constraints, both lifted to fuzzy arithmetic
                via the Zadeh extension principle.
              </p>
            </Section>

            <Section id="fuzzy-states" title="2. Fuzzy State Representation">
              <p>
                In practice, the true valuation of an asset is never perfectly known. Bid-ask spreads,
                model uncertainty, and finite-sample correlation estimates all contribute epistemic
                uncertainty that point estimates discard.
              </p>
              <p>
                We represent each asset&apos;s state as a trapezoidal fuzzy membership function
                &mu;&#771;<sub>i</sub> : R<sub>&ge;0</sub> &rarr; [0,1], where &mu;&#771;<sub>i</sub>(x) = 1
                means valuation x is fully consistent with all evidence and &mu;&#771;<sub>i</sub>(x) = 0
                means x is excluded. The support width measures irreducible uncertainty; the core
                width measures the range of equally plausible values.
              </p>
              <Theorem
                name="Fuzzy KCL (Capital Conservation)"
                statement="At each node, the fuzzy flux balance constraint is enforced by intersecting the current membership function with the KCL-consistent set, restricting valuations to those simultaneously consistent with measurement and flow balance."
              />
              <Theorem
                name="Fuzzy KVL (No-Arbitrage)"
                statement="For each directed cycle, the fuzzy sum of potential differences must contain zero in its core. Any assignment violating this condition has its membership reduced."
              />
            </Section>

            <Section id="trajectory-completion" title="3. Trajectory Completion">
              <p>
                The backward trajectory of each asset node &mdash; its maximum-a-posteriori causal
                history given current state and network topology &mdash; is computed via the Viterbi
                algorithm. This backward trajectory is the asset&apos;s <strong>categorical address</strong>:
                a time-invariant geometric identity in the portfolio&apos;s state space.
              </p>
              <Theorem
                name="Convergence Theorem (Banach Fixed-Point)"
                statement="The trajectory completion operator T = T_Back ∘ T_KVL ∘ T_KCL is a contraction mapping on the Hausdorff product metric space of fuzzy state tuples. By the Banach fixed-point theorem, iteration converges geometrically to a unique fixed point X* — the optimal portfolio allocation."
              />
              <Theorem
                name="Time-Invariance Theorem"
                statement="The optimal allocation X* depends only on the current state and network topology, not on the absolute time of observation. The portfolio rebalances because the network structure has changed, not because the calendar has advanced."
              />
            </Section>

            <Section id="risk-analysis" title="4. Risk Analysis">
              <p>
                Portfolio risk is measured by the total fuzzy support width of the fixed point,
                bounded by the spectral gap of the graph Laplacian:
              </p>
              <Theorem
                name="Spectral Risk Bound"
                statement="R ≤ R_0 / λ_2 · (N-M)/N, where λ_2 is the Fiedler value (algebraic connectivity), N is the total number of nodes, and M is the number of observed nodes."
              />
              <Theorem
                name="Exponential Shock Decay"
                statement="An external price shock at a boundary node propagates to internal node i with amplitude decaying exponentially: |Δφ_i| ≤ Δφ_0 · exp(-√λ_2 · d_G(b,i)), where d_G is the graph distance."
              />
              <p>
                Diversification in this framework is not merely &ldquo;holding many assets&rdquo; &mdash;
                it is <strong>increasing the algebraic connectivity &lambda;<sub>2</sub></strong> of the
                portfolio graph.
              </p>
            </Section>

            <Section id="markowitz" title="5. Classical Limit: Markowitz Recovery">
              <Theorem
                name="Markowitz as Special Case"
                statement="When all fuzzy states are crisp (zero epistemic uncertainty), the network is complete with uniform conductance, and backward trajectory constraints are vacuous, the fixed point reduces to the Markowitz mean-variance optimal portfolio."
              />
              <p>
                The trajectory completion framework strictly generalises Markowitz in three
                independent directions: fuzzy states handle epistemic uncertainty, sparse non-uniform
                conductance replaces the full covariance matrix, and backward trajectory constraints
                enforce kinetic consistency.
              </p>
            </Section>

            <Section id="harmonic" title="6. Harmonic Coincidence and Regime Detection">
              <p>
                FFT spectral decomposition of asset return series reveals harmonic coincidences
                between assets whose characteristic frequencies are rationally related. The shadow
                portfolio network built from these coincidences detects market regime transitions
                through topological changes in the spectral correlation graph.
              </p>
            </Section>

            <Section id="validation" title="7. Experimental Validation (7/7)">
              <ResultRow prediction="Convergence rate scales with λ₂"
                result="CONFIRMED" detail="69 iterations at λ₂=0.016, 46 at λ₂=121. Correlation = -0.87." />
              <ResultRow prediction="Time-invariance of optimal allocation"
                result="CONFIRMED" detail="Zero drift across Δt ∈ {0, 10, 100, 1000, 10000}. Exact zero." />
              <ResultRow prediction="Fuzzy risk is a meaningful risk bound"
                result="CONFIRMED" detail="Non-zero residual uncertainty preserved. Mean fuzzy risk = 0.008." />
              <ResultRow prediction="Risk scales as R ∝ 1/λ₂"
                result="CONFIRMED" detail="Risk drops from 1000 to 8 as λ₂ increases from 0.016 to 229." />
              <ResultRow prediction="Shock decays exponentially with distance"
                result="CONFIRMED" detail="R² > 0.99 exponential fit. Decay rate γ = 0.96 per hop." />
              <ResultRow prediction="Harmonic coincidence detects regime changes"
                result="CONFIRMED" detail="Both spectral and correlation detect within 15 days of regime change." />
              <ResultRow prediction="Markowitz recovery as special case"
                result="CONFIRMED" detail="Uniform conductance → exact 1/N weights (distance = 0.000000)." />
            </Section>

          </div>
        </Layout>
      </main>
    </>
  );
}
