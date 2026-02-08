use cpal::{
    InputCallbackInfo, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::{path::PathBuf, sync::mpsc, time::Duration};
use transcribe_rs::{
    TranscriptionEngine,
    engines::parakeet::{ParakeetEngine, ParakeetModelParams},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize transcription engine
    let mut engine = ParakeetEngine::new();
    engine.load_model_with_params(
        &PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"),
        ParakeetModelParams::int8(),
    )?;

    // 2. Audio capture setup
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("No input device available");

    let config: StreamConfig = input_device.default_input_config()?.into();

    // 3. Create audio buffer channel
    let (sender, receiver) = mpsc::sync_channel(1024);

    // 4. Build input stream
    let input_stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &InputCallbackInfo| {
            sender.send(data.to_vec()).unwrap();
        },
        |err| eprintln!("Error in input stream: {}", err),
        None,
    )?;

    println!("Recording audio for 5 seconds...");
    input_stream.play()?;

    // 5. Record for 5 seconds
    std::thread::sleep(Duration::from_secs(5));
    drop(input_stream); // Stop recording
    println!("Recording finished");

    // 6. Collect audio samples
    let mut audio_buffer = Vec::new();
    while let Ok(data) = receiver.try_recv() {
        audio_buffer.extend(data);
    }

    // 8. Transcribe the audio
    let result = engine.transcribe_samples(audio_buffer, None)?;
    println!("\nTranscription Result:\n{}", result.text);

    Ok(())
}
