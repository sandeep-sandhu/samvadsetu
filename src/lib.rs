//! # SamvadSetu - A library to interface with popular LLM API services
//!
//! Rust-native library for integrating with popular LLM API services.
//!
//! This library provides a simple interface for commonly used LLM services including Gemini,
//! ChatGPT, and self-hosted models such as Ollama.
//!
//! The name implies a bridge for dialogue since the library facilitates communication and
//! interaction between a programs logic and the large language models capabilities.
//! The sanskrit word saṃvāda (संवाद) refers to dialogue, and setu (सेतु) means bridge.
//!
//! # Quick Start:
//! Here is an example to quickly get started:
//!
//! //     use samvadsetu::llm::LLMTextGenerator;
//! //
//! //     let config_filename = "examples\\example1.toml";
//! //
//! //     if let Ok(app_confg) = Config::builder().add_source(config::File::new(&config_filename, FileFormat::Toml)).build() {
//! //         println!("Reading from config file: {}", config_filename);
//! //
//! //         if let Ok(llm_gen) = LLMTextGenerator::build_from_config(&app_confg, "ollama")
//! //         {
//! //             let answer1 = llm_gen.generate_text("", "How is a rainbow created in the sky? Respond very concisely.");
//! //             if let Ok(llm_response) = answer1 {
//! //                 println!("---Answer 1---\n{:?}", llm_response);
//! //                 assert_eq!(true, true);
//! //             }
//! //         }
//! //     }
//! //
//!
//!
//! By default, the api keys for chatgpt and Gemini services are picked up from environment
//! variables as given in their respective API reference pages:
//!   - ChatGPT: https://platform.openai.com/docs/api-reference/chat/create
//!   - Gemini: https://ai.google.dev/gemini-api/docs/quickstart?lang=rest
//!
//! # Configuration File
//! This example assumes the config file (.toml format) has entries like these:
//!
//! <tt>
//!
//! [llm_apis."chatgpt"]
//!
//! max_context_len = 16384
//!
//! max_gen_tokens = 8192
//!
//! temperature = 0.0
//!
//! model_name = "gpt-4o-mini"
//!
//! api_url = "https://api.openai.com/v1/chat/completions"
//!
//! model_api_timeout=200
//!
//!
//!
//! [llm_apis."gemini"]
//!
//! max_context_len = 16384
//!
//! max_gen_tokens = 8192
//!
//! temperature = 0.0
//!
//! model_name = "gemini-1.5-flash"
//!
//! api_url = "https://generativelanguage.googleapis.com/v1beta/models"
//!
//! model_api_timeout=200
//!
//! </tt>

pub mod llm;
pub mod providers {
    pub mod google;
    pub mod ollama;
    pub mod openai;
}
