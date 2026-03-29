import Link from "next/link";
import React from "react";
import Layout from "./Layout";

const Footer = () => {
  return (
    <footer className="w-full border-t-2 border-solid border-primary/20 font-medium text-lg text-light sm:text-base">
      <Layout className="py-8 flex items-center justify-between lg:flex-col lg:py-6">
        <span className="text-light/60">{new Date().getFullYear()} &copy; Fourth Stomach Framework. All Rights Reserved.</span>

        <div className="flex items-center lg:py-2 text-light/60">
          Built by&nbsp;
          <Link
            href="mailto:kundai.sachikonye@bitspark.com"
            className="underline underline-offset-2 text-primary hover:text-primaryDark transition-colors"
          >
            Kundai Farai Sachikonye
          </Link>
        </div>

        <Link
          href="/licensing"
          className="underline underline-offset-2 text-primary hover:text-primaryDark transition-colors"
        >
          License the Framework
        </Link>
      </Layout>
    </footer>
  );
};

export default Footer;
