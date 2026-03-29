import { Html, Head, Main, NextScript } from "next/document";
import Script from "next/script";

export default function Document() {
  return (
    <Html lang="en" className="dark">
      <Head />
      <body>
        <Script id="theme-lock" strategy="beforeInteractive">
          {`document.documentElement.classList.add('dark');`}
        </Script>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
