import Link from "next/link";
import type { Metadata } from "next";
import { ScanLine } from "lucide-react";
import Providers from "@/components/Providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Table Studio | Document extraction",
  description:
    "Extract, review, and correct text and tables across document pages.",
};

/** Wire page metadata and the query provider; UI and OCR languages are independent. */
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <header className="app-header">
            <div className="header-inner">
              <Link href="/" className="brand">
                <span className="brand-icon">
                  <ScanLine size={22} />
                </span>
                <span>
                  Table<span className="brand-light">Studio</span>
                </span>
              </Link>
              <span className="header-caption">DOCUMENT WORKSPACE</span>
              <span className="local-badge">
                <span className="status-dot" /> Local workspace
              </span>
            </div>
          </header>
          <main className="app-main">{children}</main>
          <footer className="app-footer">
            Full documents. Every page. Text and tables together.
          </footer>
        </Providers>
      </body>
    </html>
  );
}
