import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const EndpointCard = ({ method, path, description, params, response }) => (
  <motion.div
    className="border border-primary/20 rounded-lg bg-surface mb-4 overflow-hidden"
    initial={{ opacity: 0, y: 10 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
  >
    <div className="flex items-center gap-3 px-4 py-3 border-b border-primary/10">
      <span className={`text-xs font-bold font-mono px-2 py-1 rounded ${
        method === "GET" ? "bg-primary/20 text-primary" :
        method === "POST" ? "bg-coral/20 text-coral" :
        "bg-gold/20 text-gold"
      }`}>{method}</span>
      <code className="text-light/90 font-mono text-sm">{path}</code>
    </div>
    <div className="px-4 py-3 space-y-2">
      <p className="text-light/60 text-sm">{description}</p>
      {params && (
        <div>
          <p className="text-light/40 text-xs font-mono uppercase mb-1">Parameters</p>
          <pre className="bg-dark/50 text-primary/80 text-xs p-3 rounded font-mono overflow-x-auto">{params}</pre>
        </div>
      )}
      {response && (
        <div>
          <p className="text-light/40 text-xs font-mono uppercase mb-1">Response</p>
          <pre className="bg-dark/50 text-primaryDark/80 text-xs p-3 rounded font-mono overflow-x-auto">{response}</pre>
        </div>
      )}
    </div>
  </motion.div>
);

const Section = ({ title, children }) => (
  <div className="mb-12">
    <h2 className="text-2xl font-bold text-primary mb-6 border-b border-primary/20 pb-2">{title}</h2>
    {children}
  </div>
);

export default function ApiDocs() {
  return (
    <>
      <Head>
        <title>API Reference | Fourth Stomach</title>
        <meta name="description" content="Fourth Stomach API for programmatic access to portfolio optimisation and thermodynamic index computation." />
      </Head>
      <TransitionEffect />
      <main className="w-full mb-16 flex flex-col items-center justify-center text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="API Reference"
            className="!text-4xl mb-4 xl:!text-3xl lg:!text-2xl"
          />
          <p className="text-center text-light/50 mb-12 max-w-3xl mx-auto">
            Programmatic access to the Fourth Stomach framework. Portfolio optimisation,
            thermodynamic index computation, and circuit graph analysis.
            Requires an API key obtained through licensing.
          </p>

          <div className="max-w-4xl mx-auto">

            <div className="bg-surface border border-primary/20 rounded-lg p-6 mb-12">
              <h3 className="text-lg font-bold text-light mb-3">Authentication</h3>
              <p className="text-light/60 text-sm mb-3">All requests require an API key in the header:</p>
              <pre className="bg-dark/50 text-primary/80 text-sm p-4 rounded font-mono overflow-x-auto">
{`curl -H "Authorization: Bearer YOUR_API_KEY" \\
     https://api.fourthstomach.io/v1/portfolio/optimize`}</pre>
            </div>

            <Section title="Portfolio Optimisation">
              <EndpointCard
                method="POST"
                path="/v1/portfolio/optimize"
                description="Run trajectory completion on a portfolio circuit graph. Returns the unique fixed-point optimal allocation."
                params={`{
  "assets": [
    { "name": "AAPL", "value": 150.0, "uncertainty": 10.0 },
    { "name": "GOOGL", "value": 2800.0, "uncertainty": 50.0 }
  ],
  "couplings": [
    { "from": "AAPL", "to": "GOOGL", "conductance": 0.7 }
  ],
  "max_iterations": 500,
  "tolerance": 1e-8
}`}
                response={`{
  "fixed_point": {
    "AAPL": { "weight": 0.45, "centroid": 148.3, "support_width": 4.2 },
    "GOOGL": { "weight": 0.55, "centroid": 2790.1, "support_width": 18.7 }
  },
  "iterations": 47,
  "contraction_rate": 0.72,
  "fiedler_value": 1.4,
  "fuzzy_risk": 0.0083
}`}
              />
              <EndpointCard
                method="GET"
                path="/v1/portfolio/fiedler"
                description="Compute the Fiedler value (algebraic connectivity) of the portfolio graph."
                params={`{ "portfolio_id": "pf_abc123" }`}
                response={`{ "fiedler_value": 2.34, "spectral_gap": 1.87 }`}
              />
              <EndpointCard
                method="POST"
                path="/v1/portfolio/shock"
                description="Simulate shock propagation from a boundary node through the circuit graph."
                params={`{
  "portfolio_id": "pf_abc123",
  "shock_node": "AAPL",
  "shock_magnitude": 25.0
}`}
                response={`{
  "propagation": [
    { "node": "AAPL", "distance": 0, "amplitude": 25.0 },
    { "node": "GOOGL", "distance": 1, "amplitude": 9.55 },
    { "node": "MSFT", "distance": 2, "amplitude": 3.65 }
  ],
  "decay_rate": 0.963
}`}
              />
            </Section>

            <Section title="Distributed Thermodynamic Index">
              <EndpointCard
                method="POST"
                path="/v1/dti/compute"
                description="Compute the full thermodynamic state of a market gas from transaction data."
                params={`{
  "stocks": ["AAPL", "GOOGL", "MSFT", "AMZN"],
  "transactions": "timeseries_id_xyz",
  "window_days": 30
}`}
                response={`{
  "temperature": 5.23,
  "pressure": 0.047,
  "entropy": 1842.5,
  "free_energy": -9632.1,
  "phase": "liquid",
  "critical_distance": 0.34,
  "per_stock_entropy": {
    "AAPL": { "S_k": 0.42, "S_t": 0.61, "S_e": 0.38 },
    "GOOGL": { "S_k": 0.55, "S_t": 0.47, "S_e": 0.52 }
  }
}`}
              />
              <EndpointCard
                method="GET"
                path="/v1/dti/chemical-potential"
                description="Compute the chemical potential (true thermodynamic valuation) for each stock."
                params={`{ "market_id": "mkt_sp500" }`}
                response={`{
  "chemical_potentials": {
    "AAPL": -2.31,
    "GOOGL": -1.87,
    "TSLA": 0.45
  },
  "spontaneous_inclusions": ["AAPL", "GOOGL"],
  "spontaneous_exclusions": ["TSLA"]
}`}
              />
              <EndpointCard
                method="GET"
                path="/v1/dti/phase"
                description="Determine the current market phase (gas/liquid/crystal) and distance from critical point."
                params={`{ "market_id": "mkt_sp500" }`}
                response={`{
  "phase": "liquid",
  "T_over_Tc": 0.87,
  "gibbs_degrees_freedom": 1,
  "carnot_bound": 0.72
}`}
              />
            </Section>

            <Section title="Circuit Graph Analysis">
              <EndpointCard
                method="POST"
                path="/v1/circuit/build"
                description="Construct a portfolio circuit graph from asset data and correlation matrix."
                params={`{
  "assets": [...],
  "correlation_source": "returns_60d",
  "conductance_weights": {
    "statistical": 0.4, "economic": 0.3,
    "structural": 0.2, "information": 0.1
  }
}`}
                response={`{
  "graph_id": "cg_def456",
  "n_nodes": 50,
  "n_edges": 234,
  "fiedler_value": 3.12,
  "connected_components": 1
}`}
              />
              <EndpointCard
                method="POST"
                path="/v1/circuit/harmonic"
                description="Detect harmonic coincidences in the asset return spectra and build the shadow portfolio network."
                params={`{
  "graph_id": "cg_def456",
  "fft_window": 60,
  "tolerance": 0.05
}`}
                response={`{
  "shadow_edges": 187,
  "coincidences": [
    { "pair": ["AAPL", "MSFT"], "order": [2, 3], "correlation": 0.84 }
  ],
  "regime": "stable"
}`}
              />
            </Section>

            <div className="bg-surface border border-gold/20 rounded-lg p-6 mt-8">
              <h3 className="text-lg font-bold text-gold mb-2">Rate Limits</h3>
              <div className="grid grid-cols-3 gap-4 text-sm md:grid-cols-1">
                <div>
                  <p className="text-light/40 uppercase text-xs font-mono mb-1">Research</p>
                  <p className="text-light/80">100 requests/day</p>
                </div>
                <div>
                  <p className="text-light/40 uppercase text-xs font-mono mb-1">Commercial</p>
                  <p className="text-light/80">10,000 requests/day</p>
                </div>
                <div>
                  <p className="text-light/40 uppercase text-xs font-mono mb-1">Enterprise</p>
                  <p className="text-light/80">Unlimited</p>
                </div>
              </div>
            </div>

          </div>
        </Layout>
      </main>
    </>
  );
}
