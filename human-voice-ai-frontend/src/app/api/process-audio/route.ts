import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json(
      { success: false, error: 'Expected multipart audio upload' },
      { status: 400 }
    );
  }

  const audio = formData.get('audio');
  if (!audio || typeof audio === 'string' || audio.size === 0) {
    return NextResponse.json(
      { success: false, error: 'Provide a nonempty audio file' },
      { status: 400 }
    );
  }

  // Streamlit's health endpoint is not an inference API. Until an actual
  // speech model is connected, never infer emotion from file size or randomness.
  return NextResponse.json(
    {
      success: false,
      status: 'inference_unavailable',
      error: 'Speech emotion analysis is not connected. No emotion or confidence was calculated.'
    },
    { status: 503 }
  );
}
