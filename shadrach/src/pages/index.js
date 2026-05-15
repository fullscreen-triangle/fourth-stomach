import Head from "next/head";
import TransitionEffect from "@/components/TransitionEffect";
import dynamic from "next/dynamic";

const BullScene = dynamic(
  () => import("@/components/BullScene"),
  { ssr: false, loading: () => null }
);

export default function Home() {
  return (
    <>
      <Head>
        <title>Fourth Stomach</title>
        <meta name="description" content="Markets are gases. Portfolios are circuits." />
      </Head>
      <TransitionEffect />
      {/*
        Take up all remaining space below the Navbar.
        100dvh minus the navbar (approx 88px on desktop).
        The bull fills this area — nothing else on the page.
      */}
      <div style={{ height: 'calc(100dvh - 88px)', overflow: 'hidden' }}>
        <BullScene />
      </div>
    </>
  );
}
