import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const audioBlob = formData.get('audio') as File;
    const emotion = formData.get('emotion') as string;

    if (!audioBlob) {
      return NextResponse.json(
        { error: 'No audio file provided' },
        { status: 400 }
      );
    }

    // In a real implementation, this would connect to your Streamlit backend
    // For now, we'll simulate processing and return a response
    console.log(`Processing audio file: ${audioBlob.name}, size: ${audioBlob.size} bytes`);
    console.log(`Current emotion context: ${emotion}`);

    // Simulate some processing time
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Generate response based on emotion context
    const responses = {
      happy: "I can hear the joy in your voice! How can I help you today?",
      sad: "I sense you might be going through a tough time. I'm here to listen and help.",
      angry: "I understand you're feeling frustrated. Let's work through this together.",
      neutral: "Thank you for your message. How can I assist you?",
      excited: "Your enthusiasm is wonderful! What's got you so excited?"
    };

    const responseText = responses[emotion as keyof typeof responses] || 
                        "I received your voice message. How can I help you today?";

    return NextResponse.json({
      text: responseText,
      emotion: emotion || 'neutral',
      processing_time: '1.2s',
      confidence: Math.random() * 0.3 + 0.7 // Simulate confidence between 0.7 and 1.0
    });

  } catch (error) {
    console.error('Error processing audio:', error);
    return NextResponse.json(
      { error: 'Failed to process audio' },
      { status: 500 }
    );
  }
}
