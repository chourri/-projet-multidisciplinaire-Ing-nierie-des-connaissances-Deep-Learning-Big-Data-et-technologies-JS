import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderComponent } from '../../components/header/header';

interface Article {
  id: string;
  category: string;
  title: string;
  content: string;
  lastUpdated: string;
}

@Component({
  selector: 'app-knowledge-page',
  standalone: true,
  imports: [CommonModule, HeaderComponent],
  templateUrl: './knowledge-page.html'
})
export class KnowledgePageComponent {
  
  activeArticleId = signal('KW-01');

  articles: Article[] = [
    {
      id: 'KW-01',
      category: 'Meteorological Definitions',
      title: 'Heatwave Classification Standards',
      content: `
        <p class="mb-4">Within the context of the Beni Mellal-Khenifra region, a <strong>Heatwave Event</strong> is strictly defined by the project ontology (Rule ID: HW_DEF_01) when two specific conditions are met simultaneously:</p>
        <ul class="list-disc ml-6 space-y-2 mb-4 marker:text-slate-900">
          <li><strong>Threshold Condition:</strong> Daily Maximum Temperature ($T_{max}$) exceeds 42°C.</li>
          <li><strong>Persistence Condition:</strong> The threshold is exceeded for a minimum of 3 consecutive days ($D_{duration} \geq 3$).</li>
        </ul>
        <p>This definition aligns with the historical baseline data (1980-2020) which indicates that critical infrastructure stress begins at the 41.5°C mark in this specific topography.</p>
      `,
      lastUpdated: '2026-01-10'
    },
    {
      id: 'KW-02',
      category: 'System Architecture',
      title: 'The Neuro-Symbolic Engine',
      content: `
        <p class="mb-4">The CEEPS architecture represents a hybrid approach to Artificial Intelligence, designed to overcome the "Black Box" limitations of pure Deep Learning.</p>
        <div class="bg-slate-100 p-4 border-l-4 border-sky-800 mb-4 font-serif italic text-slate-700">
          "Deep Learning handles the noise of raw data; Knowledge Engineering handles the logic of decision making."
        </div>
        <p class="mb-4">The pipeline consists of two distinct stages:</p>
        <ol class="list-decimal ml-6 space-y-2 mb-4">
          <li><strong>Neural Stage (LSTM):</strong> A Long Short-Term Memory network processes time-series data to predict raw numerical values (Temp, Wind) for $T+7$ days.</li>
          <li><strong>Symbolic Stage (Ontology):</strong> These numerical predictions are passed to a Semantic Rule Engine which applies expert logic to infer risk levels (e.g., IF Temp > 42 THEN Alert = CRITICAL).</li>
        </ol>
      `,
      lastUpdated: '2025-12-15'
    },
    {
      id: 'KW-03',
      category: 'Data Engineering',
      title: 'Ingestion & Processing Pipeline',
      content: `
        <p class="mb-4">Data integrity is maintained through a rigid Extract-Transform-Load (ETL) process powered by Apache Spark.</p>
        <p class="mb-4"><strong>Primary Data Sources:</strong></p>
        <ul class="list-disc ml-6 space-y-2 mb-4">
          <li><strong>Sentinel-2 (ESA):</strong> Provides multi-spectral imagery for vegetation health (NDVI) monitoring.</li>
          <li><strong>Synoptic Stations:</strong> Ground-truth verification data provided by the Directorate of Meteorology (DMN).</li>
        </ul>
        <p>Missing values are imputed using KNN-based interpolation before being fed into the training set.</p>
      `,
      lastUpdated: '2026-02-01'
    }
  ];

  scrollTo(id: string) {
    this.activeArticleId.set(id);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }
}