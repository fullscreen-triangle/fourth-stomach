import AnimatedText from "@/components/AnimatedText";
import Layout from "@/components/Layout";
import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import TransitionEffect from "@/components/TransitionEffect";

const PricingCard = ({ tier, price, period, description, features, cta, highlighted }) => (
  <motion.div
    className={`p-8 rounded-xl border ${
      highlighted
        ? "border-primary bg-primary/5 shadow-glow"
        : "border-primary/20 bg-surface"
    }`}
    whileHover={{ y: -5 }}
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
  >
    {highlighted && (
      <span className="text-xs font-mono text-primary uppercase tracking-widest mb-4 block">
        Most Popular
      </span>
    )}
    <h3 className="text-2xl font-bold text-light mb-1">{tier}</h3>
    <div className="mb-4">
      <span className="text-4xl font-bold text-primary">{price}</span>
      {period && <span className="text-light/40 ml-1">{period}</span>}
    </div>
    <p className="text-light/50 text-sm mb-6">{description}</p>
    <ul className="space-y-3 mb-8">
      {features.map((f, i) => (
        <li key={i} className="flex items-start gap-2 text-sm text-light/70">
          <span className="text-primary mt-0.5">&#10003;</span>
          <span>{f}</span>
        </li>
      ))}
    </ul>
    <Link
      href="mailto:kundai.sachikonye@bitspark.com"
      className={`block text-center py-3 rounded-lg font-semibold transition-colors ${
        highlighted
          ? "bg-primary text-dark hover:bg-primaryDark"
          : "border border-primary text-primary hover:bg-primary/10"
      }`}
    >
      {cta}
    </Link>
  </motion.div>
);

const FAQItem = ({ question, answer }) => (
  <div className="border-b border-primary/10 py-4">
    <h4 className="text-light/90 font-semibold mb-2">{question}</h4>
    <p className="text-light/50 text-sm leading-relaxed">{answer}</p>
  </div>
);

export default function Licensing() {
  return (
    <>
      <Head>
        <title>Licensing | Fourth Stomach</title>
        <meta name="description" content="License the Fourth Stomach framework for research, commercial, or enterprise use." />
      </Head>
      <TransitionEffect />
      <main className="w-full mb-16 flex flex-col items-center justify-center text-light">
        <Layout className="pt-16">
          <AnimatedText
            text="License the Framework"
            className="!text-4xl mb-4 xl:!text-3xl lg:!text-2xl"
          />
          <p className="text-center text-light/50 mb-16 max-w-3xl mx-auto">
            From academic research to production deployment. Choose the license
            that matches your use case.
          </p>

          <div className="grid grid-cols-3 gap-6 mb-20 lg:grid-cols-1 max-w-5xl mx-auto">
            <PricingCard
              tier="Research"
              price="Free"
              period=""
              description="For academic research, education, and non-commercial exploration."
              features={[
                "Full paper access (LaTeX + PDF)",
                "Python validation code",
                "All experimental results (JSON/CSV)",
                "Panel figures for publications",
                "API access (100 req/day)",
                "Citation required",
                "Non-commercial use only",
              ]}
              cta="Get Research Access"
              highlighted={false}
            />
            <PricingCard
              tier="Commercial"
              price="Contact"
              period="/ year"
              description="For funds, trading firms, and fintech companies building products."
              features={[
                "Everything in Research",
                "Rust production implementation",
                "Commercial use rights",
                "API access (10,000 req/day)",
                "Priority support",
                "Custom integration guidance",
                "Sub-licensing for clients",
                "Quarterly framework updates",
              ]}
              cta="Contact for Pricing"
              highlighted={true}
            />
            <PricingCard
              tier="Enterprise"
              price="Custom"
              period=""
              description="For institutions requiring dedicated infrastructure and custom development."
              features={[
                "Everything in Commercial",
                "Unlimited API access",
                "On-premise deployment",
                "Custom system development",
                "Dedicated engineering support",
                "White-label rights",
                "SLA guarantees",
                "Joint research collaboration",
              ]}
              cta="Schedule a Call"
              highlighted={false}
            />
          </div>

          {/* What You Get */}
          <div className="max-w-4xl mx-auto mb-20">
            <h2 className="text-3xl font-bold text-center mb-10">What the Framework Includes</h2>
            <div className="grid grid-cols-2 gap-6 md:grid-cols-1">
              <motion.div className="p-6 bg-surface border border-primary/10 rounded-xl"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
                <h3 className="text-primary font-bold mb-2">7 Core Systems</h3>
                <p className="text-light/50 text-sm">CTN, STN, GCF, Multi-Modal Representations, Fuzzy Circuit Portfolio, Distributed Thermodynamic Index, Temporal Arbitrage.</p>
              </motion.div>
              <motion.div className="p-6 bg-surface border border-primary/10 rounded-xl"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
                <h3 className="text-primary font-bold mb-2">15/15 Validated Predictions</h3>
                <p className="text-light/50 text-sm">Every testable prediction confirmed. Full experimental data included in JSON/CSV format with reproducible Python scripts.</p>
              </motion.div>
              <motion.div className="p-6 bg-surface border border-primary/10 rounded-xl"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
                <h3 className="text-primary font-bold mb-2">Production Rust Engine</h3>
                <p className="text-light/50 text-sm">High-performance implementation for real-time portfolio optimisation and thermodynamic index computation. Commercial license required.</p>
              </motion.div>
              <motion.div className="p-6 bg-surface border border-primary/10 rounded-xl"
                initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
                <h3 className="text-primary font-bold mb-2">RESTful API</h3>
                <p className="text-light/50 text-sm">Programmatic access to all framework capabilities. Portfolio optimisation, DTI computation, circuit graph analysis, and harmonic detection.</p>
              </motion.div>
            </div>
          </div>

          {/* FAQ */}
          <div className="max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-10">Frequently Asked Questions</h2>
            <FAQItem
              question="Can I use the framework for academic research without a license?"
              answer="Yes. The Research tier is free for non-commercial academic use. We require citation in any resulting publications. All papers, validation code, and experimental results are freely available."
            />
            <FAQItem
              question="What is the difference between the Python validation code and the Rust engine?"
              answer="The Python code validates the mathematical claims made in the papers. It is designed for correctness and clarity, not performance. The Rust engine is the production implementation, optimised for real-time operation with market data feeds. The Rust engine requires a Commercial or Enterprise license."
            />
            <FAQItem
              question="Can I build a product on top of the framework?"
              answer="Yes, with a Commercial license. This includes the right to integrate the framework into your products and services, sub-license to your clients, and access the production Rust engine and API."
            />
            <FAQItem
              question="Do you offer custom development?"
              answer="Yes, under the Enterprise tier. This includes custom system development, dedicated engineering support, on-premise deployment, and joint research collaboration. Contact us to discuss your requirements."
            />
            <FAQItem
              question="How are updates delivered?"
              answer="Research users receive updates through the public repository. Commercial licensees receive quarterly framework updates including new systems, improved algorithms, and expanded validation results. Enterprise licensees receive continuous updates with priority access to new capabilities."
            />
          </div>

        </Layout>
      </main>
    </>
  );
}
