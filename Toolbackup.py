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
warnings.filterwarnings('ignore')

# --- Credentials & Config ---
DFSEO_USER = "admin@linkscience.ai"
DFSEO_PASS = "65573d10eab97090"
BASE_URL = "https://api.dataforseo.com/v3"

st.set_page_config(page_title="🔍 Complete SEO Insights Extractor", layout="wide")

# --- Tab Setup ---
tab1, tab2 = st.tabs(["🔍 SEO Analysis", "🎯 Smart Anchor & URL Matcher"])

# ================================
# TAB 1: ORIGINAL SEO ANALYSIS
# ================================
with tab1:
    st.title("🔍 Complete SEO Insights Extractor (DataForSEO)")
    st.markdown("Extract **Keywords**, **Backlinks**, and **Domain Rankings** for your URLs")

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
            payload = [{
                "target": domain,
                "location_name": "United States",
                "language_name": "English",
                "include_serp_info": True,
                "include_subdomains": True,
                "filters": ["keyword_properties.keyword_difficulty", ">", 0],
                "limit": 10
            }]
            
            task = dfseo_request("/dataforseo_labs/google/keywords_for_site/live", payload)
            if not task or not task.get("result"):
                return []
            
            keywords_data = []
            items = task["result"][0].get("items", [])
            
            for item in items[:5]:  # Limit to top 5 keywords
                keyword_info = item.get("keyword_info", {})
                keyword_props = item.get("keyword_properties", {})
                
                keywords_data.append({
                    "keyword": item.get("keyword", ""),
                    "search_volume": keyword_info.get("search_volume", 0),
                    "keyword_difficulty": keyword_props.get("keyword_difficulty", 0),
                    "cpc": keyword_info.get("cpc", 0),
                    "competition_level": keyword_info.get("competition_level", "")
                })
            
            return keywords_data
        except Exception as e:
            st.error(f"Keywords fetch error for {domain}: {str(e)}")
            return []

    def fetch_backlinks(domain):
        """Fetch backlinks for a domain"""
        try:
            payload = [{
                "target": domain,
                "limit": 10,
                "mode": "as_is",
                "filters": ["dofollow", "=", True]
            }]
            
            task = dfseo_request("/backlinks/backlinks/live", payload)
            if not task or not task.get("result"):
                return {}
            
            result = task["result"][0]
            items = result.get("items", [])
            
            backlinks_data = {
                "total_backlinks": result.get("total_count", 0),
                "sample_backlinks": []
            }
            
            for item in items[:3]:  # Limit to top 3 backlinks
                backlinks_data["sample_backlinks"].append({
                    "domain_from": item.get("domain_from", ""),
                    "url_from": item.get("url_from", ""),
                    "anchor": item.get("anchor", ""),
                    "domain_rank": item.get("domain_from_rank", 0),
                    "dofollow": item.get("dofollow", False)
                })
            
            return backlinks_data
        except Exception as e:
            st.error(f"Backlinks fetch error for {domain}: {str(e)}")
            return {}

    def fetch_domain_metrics(domain):
        """Fetch domain ranking and metrics"""
        try:
            # Use domain analytics whois overview for domain metrics
            payload = [{
                "limit": 1,
                "filters": [["domain", "=", domain]]
            }]
            
            task = dfseo_request("/domain_analytics/whois/overview/live", payload)
            if not task or not task.get("result"):
                return {}
            
            items = task["result"][0].get("items", [])
            if not items:
                return {}
            
            item = items[0]
            metrics = item.get("metrics", {})
            organic = metrics.get("organic", {})
            backlinks_info = item.get("backlinks_info", {})
            
            return {
                "domain_rank": organic.get("pos_1", 0) + organic.get("pos_2_3", 0) + organic.get("pos_4_10", 0),
                "organic_keywords": organic.get("count", 0),
                "organic_traffic_value": organic.get("etv", 0),
                "referring_domains": backlinks_info.get("referring_domains", 0),
                "total_backlinks": backlinks_info.get("backlinks", 0),
                "domain_authority_score": backlinks_info.get("referring_main_domains", 0)
            }
        except Exception as e:
            st.error(f"Domain metrics fetch error for {domain}: {str(e)}")
            return {}

    def process_url(url):
        """Process a single URL and extract all SEO insights"""
        domain = extract_domain(url)
        
        results = {
            "original_url": url,
            "domain": domain,
            "keywords": [],
            "backlinks": {},
            "domain_metrics": {}
        }
        
        # Fetch data from all three endpoints
        with st.spinner(f"🔍 Analyzing {domain}..."):
            # Keywords
            keywords = fetch_keywords(domain)
            results["keywords"] = keywords
            
            # Backlinks
            backlinks = fetch_backlinks(domain)
            results["backlinks"] = backlinks
            
            # Domain metrics
            domain_metrics = fetch_domain_metrics(domain)
            results["domain_metrics"] = domain_metrics
            
            # Add delay to respect rate limits
            time.sleep(1)
        
        return results

    def format_results_for_csv(all_results):
        """Format results into a flat structure suitable for CSV export"""
        formatted_data = []
        
        for result in all_results:
            row = {
                "URL": result["original_url"],
                "Domain": result["domain"],
                
                # Domain Metrics
                "Domain_Rank_Score": result["domain_metrics"].get("domain_rank", 0),
                "Organic_Keywords_Count": result["domain_metrics"].get("organic_keywords", 0),
                "Organic_Traffic_Value": result["domain_metrics"].get("organic_traffic_value", 0),
                "Referring_Domains": result["domain_metrics"].get("referring_domains", 0),
                "Total_Backlinks": result["domain_metrics"].get("total_backlinks", 0),
                "Domain_Authority_Score": result["domain_metrics"].get("domain_authority_score", 0),
                
                # Backlinks Summary
                "Total_Backlinks_Found": result["backlinks"].get("total_backlinks", 0),
                
                # Top Keywords (up to 3)
                "Top_Keyword_1": result["keywords"][0]["keyword"] if len(result["keywords"]) > 0 else "",
                "Top_Keyword_1_Volume": result["keywords"][0]["search_volume"] if len(result["keywords"]) > 0 else 0,
                "Top_Keyword_1_Difficulty": result["keywords"][0]["keyword_difficulty"] if len(result["keywords"]) > 0 else 0,
                
                "Top_Keyword_2": result["keywords"][1]["keyword"] if len(result["keywords"]) > 1 else "",
                "Top_Keyword_2_Volume": result["keywords"][1]["search_volume"] if len(result["keywords"]) > 1 else 0,
                "Top_Keyword_2_Difficulty": result["keywords"][1]["keyword_difficulty"] if len(result["keywords"]) > 1 else 0,
                
                "Top_Keyword_3": result["keywords"][2]["keyword"] if len(result["keywords"]) > 2 else "",
                "Top_Keyword_3_Volume": result["keywords"][2]["search_volume"] if len(result["keywords"]) > 2 else 0,
                "Top_Keyword_3_Difficulty": result["keywords"][2]["keyword_difficulty"] if len(result["keywords"]) > 2 else 0,
                
                # Top Backlinks (up to 2)
                "Top_Backlink_1_Domain": result["backlinks"].get("sample_backlinks", [{}])[0].get("domain_from", "") if result["backlinks"].get("sample_backlinks") else "",
                "Top_Backlink_1_URL": result["backlinks"].get("sample_backlinks", [{}])[0].get("url_from", "") if result["backlinks"].get("sample_backlinks") else "",
                "Top_Backlink_1_Anchor": result["backlinks"].get("sample_backlinks", [{}])[0].get("anchor", "") if result["backlinks"].get("sample_backlinks") else "",
                
                "Top_Backlink_2_Domain": result["backlinks"].get("sample_backlinks", [{}])[1].get("domain_from", "") if len(result["backlinks"].get("sample_backlinks", [])) > 1 else "",
                "Top_Backlink_2_URL": result["backlinks"].get("sample_backlinks", [{}])[1].get("url_from", "") if len(result["backlinks"].get("sample_backlinks", [])) > 1 else "",
                "Top_Backlink_2_Anchor": result["backlinks"].get("sample_backlinks", [{}])[1].get("anchor", "") if len(result["backlinks"].get("sample_backlinks", [])) > 1 else "",
            }
            
            formatted_data.append(row)
        
        return formatted_data

    # --- Streamlit UI for Tab 1 ---
    st.markdown("### 📤 Upload Your CSV File")
    st.markdown("Your CSV should contain a column named **'Page URL'** with the URLs to analyze.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="tab1_upload")

    if uploaded_file is not None:
        # Read the CSV
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSV loaded successfully! Found {len(df)} rows.")
            
            # Check for Page URL column
            url_column = None
            for col in df.columns:
                if 'url' in col.lower() or 'page' in col.lower():
                    url_column = col
                    break
            
            if url_column is None:
                st.error("❌ Could not find a URL column. Please ensure your CSV has a column containing URLs.")
                st.stop()
            
            st.info(f"📊 Using column: **{url_column}**")
            st.dataframe(df.head())
            
            # Process URLs
            if st.button("🚀 Start SEO Analysis", type="primary", key="start_seo"):
                urls = df[url_column].dropna().tolist()
                
                if not urls:
                    st.error("No valid URLs found in the selected column.")
                    st.stop()
                
                # Limit processing for demo (remove this limit in production)
                if len(urls) > 10:
                    st.warning(f"⚠️ Processing first 10 URLs only (found {len(urls)} total). Remove this limit in production.")
                    urls = urls[:10]
                
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                all_results = []
                
                # Process each URL
                for i, url in enumerate(urls):
                    status_text.text(f"Processing {i+1}/{len(urls)}: {url}")
                    
                    try:
                        result = process_url(url)
                        all_results.append(result)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(urls))
                        
                        # Show intermediate results
                        with results_container.container():
                            st.markdown(f"### 📊 Results for: {url}")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**🔑 Top Keywords**")
                                if result["keywords"]:
                                    for kw in result["keywords"][:3]:
                                        st.markdown(f"• {kw['keyword']} (Vol: {kw['search_volume']}, Diff: {kw['keyword_difficulty']})")
                                else:
                                    st.markdown("No keywords found")
                            
                            with col2:
                                st.markdown("**🔗 Backlinks**")
                                bl = result["backlinks"]
                                st.markdown(f"Total: {bl.get('total_backlinks', 0)}")
                                if bl.get("sample_backlinks"):
                                    for link in bl["sample_backlinks"][:2]:
                                        st.markdown(f"• {link['domain_from']}")
                            
                            with col3:
                                st.markdown("**📈 Domain Metrics**")
                                dm = result["domain_metrics"]
                                st.markdown(f"Organic Keywords: {dm.get('organic_keywords', 0)}")
                                st.markdown(f"Referring Domains: {dm.get('referring_domains', 0)}")
                                st.markdown(f"Total Backlinks: {dm.get('total_backlinks', 0)}")
                            
                            st.markdown("---")
                    
                    except Exception as e:
                        st.error(f"Error processing {url}: {str(e)}")
                        continue
                
                # Final results
                status_text.text("✅ Analysis complete!")
                
                if all_results:
                    # Format for CSV export
                    csv_data = format_results_for_csv(all_results)
                    results_df = pd.DataFrame(csv_data)
                    
                    st.markdown("### 📋 Final Results Summary")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv_output = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete SEO Analysis CSV",
                        data=csv_output,
                        file_name="seo_analysis_results.csv",
                        mime="text/csv",
                        type="primary",
                        key="download_seo"
                    )
                    
                    # Show summary statistics
                    st.markdown("### 📊 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_organic = results_df['Organic_Keywords_Count'].mean()
                        st.metric("Avg Organic Keywords", f"{avg_organic:.0f}")
                    
                    with col2:
                        avg_backlinks = results_df['Total_Backlinks'].mean()
                        st.metric("Avg Total Backlinks", f"{avg_backlinks:.0f}")
                    
                    with col3:
                        avg_referring = results_df['Referring_Domains'].mean()
                        st.metric("Avg Referring Domains", f"{avg_referring:.0f}")
                    
                    with col4:
                        domains_analyzed = len(results_df)
                        st.metric("Domains Analyzed", domains_analyzed)
                
                else:
                    st.error("❌ No results were generated. Please check your URLs and try again.")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

    else:
        st.markdown("""
        ### 📝 Instructions:
        1. **Upload a CSV file** containing URLs in a column named 'Page URL' (or similar)
        2. **Click 'Start SEO Analysis'** to begin processing
        3. **Wait for results** - each URL will be analyzed for:
           - 🔑 **Keywords**: Top ranking keywords with search volume and difficulty
           - 🔗 **Backlinks**: Total backlinks and sample high-quality links
           - 📈 **Domain Metrics**: Domain authority, organic traffic, and ranking data
        4. **Download the results** as a comprehensive CSV file
        
        ### 🔧 Features:
        - ✅ **Multi-source analysis** using DataForSEO APIs
        - ✅ **Real-time progress tracking**
        - ✅ **Comprehensive metrics** export
        - ✅ **Error handling** and retry logic
        - ✅ **Rate limiting** to respect API limits
        
        ### ⚠️ Note:
        This tool uses DataForSEO APIs which have usage costs. Monitor your API usage accordingly.
        """)

# ================================
# TAB 2: SMART ANCHOR & URL MATCHER
# ================================
with tab2:
    st.title("🎯 Smart Anchor & URL Matcher")
    st.markdown("Analyze content similarity between source pages and client pages to suggest optimal anchor texts and URL matches")

    # Content extraction functions
    def extract_content_from_url(url, timeout=10):
        """Extract text content from a URL"""
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length to avoid memory issues
            return text[:5000] if text else ""
            
        except Exception as e:
            return f"Error extracting content: {str(e)}"

    def preprocess_text(text):
        """Clean and preprocess text for analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def calculate_similarity(text1, text2):
        """Calculate cosine similarity between two texts"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Preprocess texts
            text1_clean = preprocess_text(text1)
            text2_clean = preprocess_text(text2)
            
            if not text1_clean or not text2_clean:
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            return 0.0

    def extract_keywords_from_text(text, num_keywords=10):
        """Extract important keywords from text using TF-IDF"""
        try:
            if not text:
                return []
            
            text_clean = preprocess_text(text)
            if not text_clean:
                return []
            
            vectorizer = TfidfVectorizer(
                max_features=num_keywords,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            tfidf_matrix = vectorizer.fit_transform([text_clean])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords with scores
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw[0] for kw in keyword_scores if kw[1] > 0]
            
        except Exception as e:
            return []

    def generate_anchor_suggestions(source_content, client_content, existing_anchor=""):
        """Generate anchor text suggestions based on content analysis"""
        suggestions = []
        
        # Extract keywords from both contents
        source_keywords = extract_keywords_from_text(source_content, 15)
        client_keywords = extract_keywords_from_text(client_content, 15)
        
        # Find common keywords
        common_keywords = list(set(source_keywords) & set(client_keywords))
        
        # Generate suggestions based on content overlap
        if common_keywords:
            # Use top common keywords
            for keyword in common_keywords[:3]:
                suggestions.append(keyword.title())
            
            # Create phrase combinations
            if len(common_keywords) >= 2:
                suggestions.append(f"{common_keywords[0]} {common_keywords[1]}".title())
        
        # Add client-focused suggestions
        if client_keywords:
            suggestions.extend([kw.title() for kw in client_keywords[:2]])
        
        # Generic relevant suggestions
        generic_suggestions = [
            "Learn More",
            "Read More",
            "Check This Out",
            "Visit Site",
            "Explore Here"
        ]
        
        # Add some generic options
        suggestions.extend(generic_suggestions[:2])
        
        # Remove duplicates and clean up
        suggestions = list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order
        suggestions = [s for s in suggestions if s and len(s.strip()) > 0]
        
        # Limit to 5 suggestions
        return suggestions[:5]

    def analyze_anchor_opportunities(df, source_url_col, client_url_col, anchor_col, max_rows=None, min_similarity=0.1):
        """Analyze anchor opportunities with content similarity"""
        
        if max_rows:
            df = df.head(max_rows)
        
        results = []
        total_rows = len(df)
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, row in df.iterrows():
            status_text.text(f"Analyzing {i+1}/{total_rows}: {row[source_url_col]}")
            
            try:
                source_url = row[source_url_col]
                client_url = row[client_url_col]
                existing_anchor = str(row[anchor_col]) if anchor_col and not pd.isna(row[anchor_col]) else ""
                
                # Extract content from both URLs
                source_content = extract_content_from_url(source_url)
                client_content = extract_content_from_url(client_url)
                
                # Calculate similarity
                similarity_score = calculate_similarity(source_content, client_content)
                
                # Generate anchor suggestions
                anchor_suggestions = generate_anchor_suggestions(source_content, client_content, existing_anchor)
                
                # Determine quality assessment
                if similarity_score >= 0.5:
                    quality = "High"
                elif similarity_score >= 0.2:
                    quality = "Medium"
                else:
                    quality = "Low"
                
                result = {
                    'source_url': source_url,
                    'client_url': client_url,
                    'existing_anchor': existing_anchor,
                    'similarity_score': similarity_score,
                    'quality': quality,
                    'suggested_anchors': anchor_suggestions,
                    'source_content_preview': source_content[:200] + "..." if len(source_content) > 200 else source_content,
                    'client_content_preview': client_content[:200] + "..." if len(client_content) > 200 else client_content,
                    'content_extracted': len(source_content) > 50 and len(client_content) > 50
                }
                
                # Only include results above minimum similarity threshold
                if similarity_score >= min_similarity:
                    results.append(result)
                
                # Update progress
                progress_bar.progress((i + 1) / total_rows)
                
                # Small delay to prevent overwhelming servers
                time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Error analyzing row {i+1}: {str(e)}")
                continue
        
        status_text.text("✅ Analysis complete!")
        return results

    # UI for Tab 2
    st.markdown("### 📤 Upload Your Anchor Opportunities CSV")
    st.markdown("Your CSV should contain columns for **Source URL**, **Client URL**, and optionally **Anchor Text**")

    uploaded_file_tab2 = st.file_uploader("Choose your anchor opportunities CSV file", type="csv", key="tab2_upload")

    if uploaded_file_tab2 is not None:
        try:
            df_anchors = pd.read_csv(uploaded_file_tab2)
            st.success(f"✅ CSV loaded successfully! Found {len(df_anchors)} rows.")
            
            # Show column selection
            st.markdown("### 🎯 Column Mapping")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                source_url_col = st.selectbox("Source URL Column", df_anchors.columns, key="source_col")
            
            with col2:
                client_url_col = st.selectbox("Client URL Column", df_anchors.columns, key="client_col")
            
            with col3:
                anchor_col = st.selectbox("Anchor Column (Optional)", ["None"] + list(df_anchors.columns), key="anchor_col")
                if anchor_col == "None":
                    anchor_col = None
            
            # Configuration options
            st.markdown("### ⚙️ Analysis Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                max_rows = st.number_input("Max rows to process", min_value=1, max_value=100, value=10, help="Limit processing for faster results")
            
            with col2:
                min_similarity = st.slider("Minimum similarity threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Filter out low-quality matches")
            
            # Preview data
            st.markdown("### 📊 Data Preview")
            st.dataframe(df_anchors.head())
            
            # Start analysis
            if st.button("🚀 Start Anchor Analysis", type="primary", key="start_anchor"):
                if not source_url_col or not client_url_col:
                    st.error("Please select both Source URL and Client URL columns.")
                    st.stop()
                
                # Run analysis
                results = analyze_anchor_opportunities(
                    df_anchors, 
                    source_url_col, 
                    client_url_col, 
                    anchor_col, 
                    max_rows, 
                    min_similarity
                )
                
                if results:
                    # Convert results to DataFrame for display
                    results_df = pd.DataFrame(results)
                    
                    # Display summary metrics
                    st.markdown("### 📊 Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Analyzed", len(results))
                    
                    with col2:
                        avg_similarity = results_df['similarity_score'].mean()
                        st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                    
                    with col3:
                        high_quality = len(results_df[results_df['quality'] == 'High'])
                        st.metric("High Quality Matches", high_quality)
                    
                    with col4:
                        success_rate = len(results_df[results_df['content_extracted'] == True]) / len(results_df) * 100
                        st.metric("Content Extraction Success", f"{success_rate:.1f}%")
                    
                    # Filter options
                    st.markdown("### 🔍 Filter Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        quality_filter = st.multiselect("Quality Filter", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
                    
                    with col2:
                        similarity_filter = st.slider("Minimum Similarity for Display", 0.0, 1.0, 0.0, 0.05)
                    
                    # Apply filters
                    filtered_df = results_df[
                        (results_df['quality'].isin(quality_filter)) & 
                        (results_df['similarity_score'] >= similarity_filter)
                    ]
                    
                    # Display results
                    st.markdown(f"### 📋 Results ({len(filtered_df)} matches)")
                    
                    if len(filtered_df) > 0:
                        # Display results in expandable format
                        for idx, row in filtered_df.iterrows():
                            quality_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                            
                            with st.expander(f"{quality_emoji[row['quality']]} Similarity: {row['similarity_score']:.3f} | {row['source_url'][:50]}..."):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Source URL:**")
                                    st.write(row['source_url'])
                                    st.markdown("**Source Content Preview:**")
                                    st.write(row['source_content_preview'])
                                
                                with col2:
                                    st.markdown("**Client URL:**")
                                    st.write(row['client_url'])
                                    st.markdown("**Client Content Preview:**")
                                    st.write(row['client_content_preview'])
                                
                                st.markdown("---")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("**Existing Anchor:**")
                                    st.write(row['existing_anchor'] if row['existing_anchor'] else "None")
                                
                                with col2:
                                    st.markdown("**Quality Assessment:**")
                                    st.write(f"{quality_emoji[row['quality']]} {row['quality']}")
                                
                                with col3:
                                    st.markdown("**Similarity Score:**")
                                    st.write(f"{row['similarity_score']:.3f}")
                                
                                st.markdown("**🎯 Suggested Anchor Texts:**")
                                for i, anchor in enumerate(row['suggested_anchors'], 1):
                                    st.write(f"{i}. {anchor}")
                                
                                # Recommendations based on similarity
                                if row['similarity_score'] >= 0.5:
                                    st.success("💡 **Recommendation**: Excellent match! This is a high-priority linking opportunity.")
                                elif row['similarity_score'] >= 0.2:
                                    st.info("💡 **Recommendation**: Good match. Consider this opportunity for linking.")
                                else:
                                    st.warning("💡 **Recommendation**: Low similarity. Consider reviewing client page selection.")
                        
                        # Prepare data for CSV export
                        export_data = []
                        for _, row in filtered_df.iterrows():
                            export_row = {
                                'Source_URL': row['source_url'],
                                'Client_URL': row['client_url'],
                                'Existing_Anchor': row['existing_anchor'],
                                'Similarity_Score': row['similarity_score'],
                                'Quality_Assessment': row['quality'],
                                'Suggested_Anchor_1': row['suggested_anchors'][0] if len(row['suggested_anchors']) > 0 else '',
                                'Suggested_Anchor_2': row['suggested_anchors'][1] if len(row['suggested_anchors']) > 1 else '',
                                'Suggested_Anchor_3': row['suggested_anchors'][2] if len(row['suggested_anchors']) > 2 else '',
                                'Suggested_Anchor_4': row['suggested_anchors'][3] if len(row['suggested_anchors']) > 3 else '',
                                'Suggested_Anchor_5': row['suggested_anchors'][4] if len(row['suggested_anchors']) > 4 else '',
                                'Source_Content_Preview': row['source_content_preview'],
                                'Client_Content_Preview': row['client_content_preview'],
                                'Content_Extraction_Success': row['content_extracted']
                            }
                            export_data.append(export_row)
                        
                        export_df = pd.DataFrame(export_data)
                        
                        # Download button
                        csv_output = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Enhanced Anchor Analysis CSV",
                            data=csv_output,
                            file_name="anchor_analysis_results.csv",
                            mime="text/csv",
                            type="primary",
                            key="download_anchor"
                        )
                        
                        # Additional insights
                        st.markdown("### 💡 Key Insights")
                        
                        high_sim_count = len(filtered_df[filtered_df['similarity_score'] >= 0.5])
                        medium_sim_count = len(filtered_df[(filtered_df['similarity_score'] >= 0.2) & (filtered_df['similarity_score'] < 0.5)])
                        low_sim_count = len(filtered_df[filtered_df['similarity_score'] < 0.2])
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🟢 High Priority", high_sim_count, help="Similarity ≥ 0.5")
                        
                        with col2:
                            st.metric("🟡 Medium Priority", medium_sim_count, help="Similarity 0.2-0.5")
                        
                        with col3:
                            st.metric("🔴 Low Priority", low_sim_count, help="Similarity < 0.2")
                        
                        # Recommendations
                        st.markdown("### 🎯 Strategic Recommendations")
                        
                        if high_sim_count > 0:
                            st.success(f"✅ **{high_sim_count} high-priority opportunities** identified. These should be your primary focus for link building.")
                        
                        if medium_sim_count > 0:
                            st.info(f"📊 **{medium_sim_count} medium-priority opportunities** found. These are worth pursuing after high-priority links.")
                        
                        if low_sim_count > 0:
                            st.warning(f"⚠️ **{low_sim_count} low-similarity matches** detected. Consider reviewing client page selection for these opportunities.")
                        
                        # Best practices
                        st.markdown("### 📚 Best Practices")
                        st.markdown("""
                        - **Focus on high-similarity matches** (≥0.5) for maximum relevance and SEO value
                        - **Use suggested anchor texts** that reflect content overlap between pages
                        - **Review low-similarity matches** - they might need different client pages
                        - **Test different anchor variations** to find what works best for each context
                        - **Monitor performance** of implemented links to refine future strategies
                        """)
                    
                    else:
                        st.warning("No results match your current filters. Try adjusting the similarity threshold or quality filters.")
                
                else:
                    st.error("❌ No results were generated. Please check your URLs and configuration.")
        
        except Exception as e:
            st.error(f"❌ Error processing CSV file: {str(e)}")
    
    else:
        st.markdown("""
        ### 📝 Instructions for Smart Anchor Analysis:
        
        1. **Upload a CSV file** with your anchor opportunities containing:
           - **Source URL column**: The pages where you want to place links
           - **Client URL column**: Your client's pages to link to
           - **Anchor column** (optional): Existing anchor text suggestions
        
        2. **Configure analysis settings**:
           - **Max rows**: Limit processing for testing (increase for full analysis)
           - **Similarity threshold**: Filter out low-quality matches
        
        3. **Review intelligent suggestions**:
           - 🧠 **Content analysis** using TF-IDF and cosine similarity
           - 🎯 **Contextual anchor suggestions** based on content overlap
           - 📊 **Quality scoring** to prioritize opportunities
           - 💡 **Strategic recommendations** for implementation
        
        ### 🔧 How It Works:
        - **Content Extraction**: Scrapes and analyzes text from both source and client pages
        - **Similarity Analysis**: Uses machine learning to calculate content relevance (0-1 scale)
        - **Anchor Generation**: Creates contextually relevant anchor text suggestions
        - **Quality Assessment**: Categorizes opportunities as High/Medium/Low priority
        - **Strategic Insights**: Provides actionable recommendations for link building
        
        ### 💡 Use Cases:
        - **Content Matching**: Find the best client pages for each linking opportunity
        - **Anchor Optimization**: Get data-driven anchor text suggestions
        - **Quality Control**: Identify and prioritize high-value linking opportunities
        - **Strategy Planning**: Make informed decisions about link building campaigns
        
        ### ⚠️ Notes:
        - Processing time depends on the number of URLs and server response times
        - Content extraction works best with publicly accessible pages
        - Some sites may block automated content extraction
        - Results are based on current page content and may change over time
        """)

# Add footer
st.markdown("---")
st.markdown("### 🔧 Tool Information")
st.markdown("""
**Tab 1: SEO Analysis** - Comprehensive keyword, backlink, and domain analysis using DataForSEO APIs  
**Tab 2: Smart Anchor Matcher** - AI-powered content similarity analysis for optimal anchor text suggestions

*Built with Streamlit, scikit-learn, and advanced NLP techniques*
""")