import { Component, input } from '@angular/core';
import { MeteoData } from '../../models/weather.interface';

@Component({
  selector: 'app-alert-list',
  imports: [],
  templateUrl: './alert-list.html',
  styleUrl: './alert-list.css',
})
export class AlertListComponent {
  data = input<MeteoData[]>([]);
}
