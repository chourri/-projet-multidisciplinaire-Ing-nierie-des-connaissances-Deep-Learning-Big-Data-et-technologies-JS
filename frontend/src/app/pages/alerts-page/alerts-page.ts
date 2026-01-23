import { Component, inject, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { toSignal } from '@angular/core/rxjs-interop';
import { WeatherService } from '../../services/weather';
import { HeaderComponent } from '../../components/header/header';

@Component({
  selector: 'app-alerts-page',
  standalone: true,
  imports: [CommonModule, HeaderComponent],
  templateUrl: './alerts-page.html'
})
export class AlertsPageComponent {
  private weatherService = inject(WeatherService);
  forecastSignal = toSignal(this.weatherService.getForecast(), { initialValue: null });
  weatherData = computed(() => this.forecastSignal()?.data || []);

  downloadCSV() {
    const data = this.weatherData();
    if (!data.length) return;

    const headers = ['Date', 'Max Temp (C)', 'Humidity (%)', 'Wind (km/h)', 'Alert Type', 'Alert Level'];
    const rows = data.map(day => [
      day.date,
      day.tmax,
      day.rhum,
      day.wspd,
      day.alert ? day.alert.type : 'Normal',
      day.alert ? day.alert.level : 'None'
    ]);

    const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `CEEPS_Registry_Export_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  // Track which date is currently expanded
  expandedDate = signal<string | null>(null);

  toggleDetails(date: string) {
    if (this.expandedDate() === date) {
      this.expandedDate.set(null); // Close if already open
    } else {
      this.expandedDate.set(date); // Open new one
    }
  }

  // Helper to generate professional-looking "Reasoning" text based on alert type
  getReasoning(alertType: string, temp: number): string {
    if (alertType.includes('HEAT')) {
      return `RULE_ID_404: T_MAX (${temp}°C) > REGIONAL_THRESHOLD (42°C) AND PERSISTENCE_FORECAST > 48H.`;
    }
    if (alertType.includes('WIND')) {
      return `RULE_ID_102: WIND_VELOCITY > 75km/h AND DIRECTION_VARIANCE < 10°.`;
    }
    return `ANOMALY_DETECTED: DEVIATION FROM BASELINE > 2.5 STD_DEV.`;
  }
}