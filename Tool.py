import streamlit as st
import pandas as pd
import requests
import time
from urllib.parse import urlparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from bs4 import BeautifulSoup
import warnings

# --- Page and Global Configuration ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="🔍 Complete SEO Insights Extractor", layout="wide")

# --- Credentials & Config (for Tab 1) ---
DFSEO_USER = "admin@linkscience.ai"
DFSEO_PASS = "65573d10eab97090"
BASE_URL = "https://api.dataforseo.com/v3"

# --- Tab Setup ---
tab1, tab2, tab3 = st.tabs(["🔍 SEO Analysis", "🎯 Smart Anchor & URL Matcher", "📖 About / How To Use"])

# ================================
# TAB 1: ORIGINAL SEO ANALYSIS (Unchanged)
# ================================
with tab1:
    st.title("🔍 Complete SEO Insights Extractor (DataForSEO)")
    st.markdown("Extract **Keywords**, **Backlinks**, and **Domain Rankings** for your URLs")

    # --- Helper Functions for Tab 1 ---
    def extract_domain(url):
        """Extract domain from URL"""
        try:
            parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'http://{url}')
            return parsed.netloc.replace('www.', '')
        except:
            return url

    def dfseo_request(endpoint, payload, max_retries=3):
        """Generic DataForSEO API request function"""
        url = BASE_URL + endpoint
        auth = (DFSEO_USER, DFSEO_PASS)
        for attempt in range(max_retries):
            try:
                response = requests.post(url, auth=auth, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status_code") == 20000:
                        return data["tasks"][0] if data.get("tasks") else None
                    else:
                        st.warning(f"API Warning: {data.get('status_message', 'Unknown error')}")
                time.sleep(1)
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Request failed after {max_retries} attempts: {str(e)}")
        return None

    def fetch_keywords(domain):
        """Fetch keywords for a domain"""
        try:
            payload = [{"target": domain, "location_name": "United States", "language_name": "English", "limit": 5}]
            task = dfseo_request("/dataforseo_labs/google/keywords_for_site/live", payload)
            if not task or not task.get("result"): return []
            return [item for item in task["result"][0].get("items", [])]
        except Exception: return []

    def fetch_backlinks(domain):
        """Fetch backlinks for a domain"""
        try:
            payload = [{"target": domain, "limit": 3, "mode": "as_is"}]
            task = dfseo_request("/backlinks/backlinks/live", payload)
            if not task or not task.get("result"): return {}
            return task["result"][0]
        except Exception: return {}

    def fetch_domain_metrics(domain):
        """Fetch domain ranking and metrics"""
        try:
            payload = [{"targets": [domain]}]
            task = dfseo_request("/dataforseo_labs/google/domain_rank_overview/live", payload)
            if not task or not task.get("result"): return {}
            return task["result"][0].get("items", [{}])[0]
        except Exception: return {}

    def process_url(url):
        """Process a single URL and extract all SEO insights"""
        domain = extract_domain(url)
        results = {"original_url": url, "domain": domain}
        with st.spinner(f"🔍 Analyzing {domain}..."):
            results["keywords"] = fetch_keywords(domain)
            results["backlinks"] = fetch_backlinks(domain)
            results["domain_metrics"] = fetch_domain_metrics(domain)
            time.sleep(1)
        return results

    def format_results_for_csv(all_results):
        """Format results into a flat structure suitable for CSV export"""
        formatted_data = []
        for res in all_results:
            row = {
                "URL": res["original_url"],
                "Domain": res["domain"],
                "Domain Rank": res["domain_metrics"].get("rank", 0),
                "Total Backlinks": res["backlinks"].get("total_count", 0),
            }
            # Add top keywords
            for i in range(3):
                kw = res["keywords"][i] if i < len(res["keywords"]) else {}
                row[f"Top Keyword {i+1}"] = kw.get("keyword", "")
                row[f"Keyword {i+1} Volume"] = kw.get("keyword_info", {}).get("search_volume", 0)
            # Add top backlinks
            for i in range(2):
                bl = res["backlinks"].get("items", [])
                link = bl[i] if i < len(bl) else {}
                row[f"Top Backlink {i+1} Domain"] = link.get("domain_from", "")
                row[f"Top Backlink {i+1} Anchor"] = link.get("anchor", "")
            formatted_data.append(row)
        return formatted_data

    # --- UI for Tab 1 ---
    st.markdown("### 📤 Upload Your CSV File")
    st.markdown("Your CSV should contain a column with the URLs to analyze.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="tab1_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ CSV loaded successfully! Found {len(df)} rows.")
        url_column = st.selectbox("Select the column containing URLs", df.columns, key="tab1_url_col")
        st.dataframe(df.head())
        
        if st.button("🚀 Start SEO Analysis", type="primary", key="start_seo"):
            urls = df[url_column].dropna().unique().tolist()
            if len(urls) > 10:
                st.warning(f"⚠️ Found {len(urls)} URLs. Processing the first 10 for this demo.")
                urls = urls[:10]

            progress_bar = st.progress(0)
            status_text = st.empty()
            all_results = []

            for i, url in enumerate(urls):
                status_text.text(f"Processing {i+1}/{len(urls)}: {url}")
                all_results.append(process_url(url))
                progress_bar.progress((i + 1) / len(urls))

            status_text.text("✅ Analysis complete!")
            if all_results:
                results_df = pd.DataFrame(format_results_for_csv(all_results))
                st.markdown("### 📋 Final Results Summary")
                st.dataframe(results_df)
                csv_output = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", csv_output, "seo_analysis.csv", "text/csv", type="primary")

# ====================================================
# TAB 2: SMART ANCHOR & URL MATCHER (Corrected Logic)
# ====================================================
with tab2:
    st.title("🎯 Smart Anchor & URL Matcher")
    st.markdown("""
    **Find the most relevant client page for any linking opportunity.** This tool analyzes the content of a source page and compares it against a list of your client's pages to recommend the best match and suggest ideal anchor texts.
    """)

    # --- Content extraction and analysis functions ---
    @st.cache_data(ttl=3600) # Cache content for 1 hour
    def extract_content_from_url(url, timeout=10):
        """Extracts and cleans text content from a URL."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return ' '.join(text.split())[:7000] # Limit content length
        except Exception as e:
            return None

    def preprocess_text(text):
        """Cleans text for TF-IDF analysis."""
        if not isinstance(text, str): return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        return ' '.join(text.split())

    def calculate_similarity(text1_clean, text2_clean):
        """Calculates cosine similarity between two preprocessed texts."""
        try:
            if not text1_clean or not text2_clean: return 0.0
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
            return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            return 0.0

    @st.cache_data
    def extract_keywords_from_text(text, num_keywords=10):
        """Extracts top keywords using TF-IDF."""
        try:
            if not text: return []
            clean_text = preprocess_text(text)
            if not clean_text: return []
            vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english', ngram_range=(1, 3))
            vectorizer.fit_transform([clean_text])
            return vectorizer.get_feature_names_out().tolist()
        except Exception:
            return []

    def generate_anchor_suggestions(source_content, client_content):
        """Generates anchor text suggestions."""
        source_keywords = extract_keywords_from_text(source_content, 15)
        client_keywords = extract_keywords_from_text(client_content, 15)
        common_keywords = list(set(source_keywords) & set(client_keywords))
        
        suggestions = [kw.title() for kw in common_keywords[:3]]
        suggestions.extend([kw.title() for kw in client_keywords[:2] if kw.title() not in suggestions])
        suggestions.extend(["Learn More", "Read More Here", "Visit Our Website"])
        
        return list(dict.fromkeys(suggestions))[:5] # Return unique suggestions

    # --- Core Matching Logic ---
    def find_best_client_page_matches(source_df, source_url_col, client_urls, max_rows=None):
        """For each source URL, finds the best matching client URL."""
        results = []
        
        st.info(f"Analyzing content of {len(client_urls)} client pages. This may take a moment...")
        client_pages_content = []
        for url in client_urls:
            content = extract_content_from_url(url)
            if content:
                client_pages_content.append({"url": url, "content": content, "clean_content": preprocess_text(content)})
        
        if not client_pages_content:
            st.error("Could not extract content from any client URLs. Please check the URLs.")
            return []
        st.success(f"Successfully analyzed {len(client_pages_content)} client pages.")

        if max_rows: source_df = source_df.head(max_rows)
        
        total_rows = len(source_df)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, row in source_df.iterrows():
            source_url = row[source_url_col]
            status_text.text(f"Analyzing Source URL ({i+1}/{total_rows}): {source_url}")
            
            source_content = extract_content_from_url(source_url)
            if not source_content:
                st.warning(f"Skipping source URL {source_url}: Could not extract content.")
                continue

            source_content_clean = preprocess_text(source_content)
            best_match = {"client_url": None, "similarity_score": -1.0, "client_content": None}

            for client_page in client_pages_content:
                similarity = calculate_similarity(source_content_clean, client_page['clean_content'])
                if similarity > best_match['similarity_score']:
                    best_match.update({
                        'similarity_score': similarity,
                        'client_url': client_page['url'],
                        'client_content': client_page['content']
                    })

            if best_match['client_url']:
                if best_match['similarity_score'] >= 0.4: quality = "High"
                elif best_match['similarity_score'] >= 0.15: quality = "Medium"
                else: quality = "Low"

                results.append({
                    'Source URL': source_url,
                    'Suggested Client URL': best_match['client_url'],
                    'Similarity Score': round(best_match['similarity_score'], 3),
                    'Match Quality': quality,
                    'Suggested Anchors': ", ".join(generate_anchor_suggestions(source_content, best_match['client_content'])),
                })
            
            progress_bar.progress((i + 1) / total_rows)
            time.sleep(0.2) 

        status_text.text("✅ Analysis complete!")
        return results

    # --- UI for Tab 2 ---
    st.markdown("### 1. Provide Source URLs")
    st.markdown("Upload a CSV file containing the potential linking pages (e.g., from Ahrefs, Semrush, or your own prospecting list). The file must have a header.")
    uploaded_file_tab2 = st.file_uploader("Upload Source URLs CSV", type="csv", key="tab2_source_upload")

    st.markdown("### 2. Provide Client URLs")
    st.markdown("Paste the key URLs from your client's site you want to build links to (one URL per line).")
    client_urls_input = st.text_area("Client Site URLs", height=150, placeholder="https://www.client.com/page-a\nhttps://www.client.com/page-b\nhttps://www.client.com/service-c")

    if uploaded_file_tab2 and client_urls_input:
        try:
            df_sources = pd.read_csv(uploaded_file_tab2)
            client_urls = [url.strip() for url in client_urls_input.split('\n') if url.strip()]

            if df_sources.empty or not client_urls:
                st.warning("Please ensure the source CSV is not empty and provide at least one client URL.")
            else:
                st.success(f"✅ Source CSV loaded ({len(df_sources)} rows) and {len(client_urls)} client URLs provided.")
                
                col1, col2 = st.columns(2)
                source_url_col = col1.selectbox("Select the Source URL Column", df_sources.columns, key="tab2_url_col")
                max_rows = col2.number_input("Max source rows to process", 1, len(df_sources), min(20, len(df_sources)))
                
                if st.button("🚀 Find Best URL Matches & Anchors", type="primary", key="start_url_matching"):
                    results = find_best_client_page_matches(df_sources, source_url_col, client_urls, max_rows)

                    if results:
                        results_df = pd.DataFrame(results)
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")

                        col1_res, col2_res, col3_res = st.columns(3)
                        avg_sim = results_df['Similarity Score'].mean()
                        col1_res.metric("Avg. Similarity", f"{avg_sim:.2f}")
                        high_q_count = (results_df['Match Quality'] == 'High').sum()
                        col2_res.metric("High-Quality Matches", high_q_count)
                        med_q_count = (results_df['Match Quality'] == 'Medium').sum()
                        col3_res.metric("Medium-Quality Matches", med_q_count)
                        
                        st.dataframe(results_df)
                        
                        csv_output = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Match Results as CSV",
                            data=csv_output,
                            file_name="smart_anchor_matches.csv",
                            mime="text/csv",
                            type="primary"
                        )
                    else:
                        st.error("No matches could be generated. This might be due to issues accessing the content of the provided URLs.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# ================================
# TAB 3: ABOUT / HOW TO USE (New)
# ================================
with tab3:
    st.title("📖 About & How To Use This Tool")
    
    st.markdown("---")

    st.header("🔍 SEO Analysis")
    st.markdown("""
    This tool provides a high-level overview of the SEO profile for a list of URLs using the DataForSEO API.
    
    **How to use:**
    1.  Prepare a CSV file with a list of URLs you want to analyze. Make sure the file has a header row.
    2.  Upload the CSV file using the uploader.
    3.  Select the column from your file that contains the URLs.
    4.  Click **"Start SEO Analysis"**.
    
    **What you get:**
    - **Domain Rank**: The overall authority of the domain.
    - **Total Backlinks**: The total number of backlinks pointing to the domain.
    - **Top Keywords**: The top organic keywords the domain ranks for, along with their search volume.
    - **Top Backlinks**: A sample of the most powerful backlinks, showing the linking domain and anchor text.
    - You can download the full results as a CSV file.
    """)

    st.markdown("---")

    st.header("🎯 Smart Anchor & URL Matcher")
    st.markdown("""
    This is an advanced tool for link builders. It solves a common problem: "I have a list of websites where I could get a link, but which page on my client's site is the best one to link to, and what anchor text should I use?"
    
    **How to use:**
    1.  **Provide Source URLs**: Upload a CSV file containing potential linking opportunities. This could be a list of prospect sites, guest post targets, etc. The file needs a column containing the specific URL of the page where you might get a link.
    2.  **Provide Client URLs**: In the text box, paste the most important URLs from your client's website that you want to build links to. List one URL per line.
    3.  Select the correct **Source URL Column** from your uploaded CSV.
    4.  Choose the maximum number of rows you want to process (useful for quick tests).
    5.  Click **"Find Best URL Matches & Anchors"**.
    
    **What it does:**
    - The tool visits each Source URL and analyzes its text content.
    - It then compares this content to the content of *every* Client URL you provided.
    - Using TF-IDF and Cosine Similarity, it calculates a **relevance score** between the source page and each client page.
    
    **What you get:**
    - **Suggested Client URL**: For each source URL, the tool recommends the single best client URL that matches its content.
    - **Similarity Score**: A score from 0 to 1 indicating how relevant the two pages are. Higher is better.
    - **Match Quality**: A simple "High," "Medium," or "Low" rating to help you prioritize.
    - **Suggested Anchors**: A list of 5 contextually relevant anchor text ideas, generated by finding common keywords between the two pages.
    - The final results can be downloaded as a CSV for your outreach campaigns.
    """)
    
# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>All rights Reserved 2025 by White Light Digital Marketing</div>", unsafe_allow_html=True)
