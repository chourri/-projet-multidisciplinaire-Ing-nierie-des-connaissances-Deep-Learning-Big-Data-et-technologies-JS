import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet], // RouterOutlet must be imported here
  template: `<router-outlet></router-outlet>`
})
export class App {
  protected readonly title = signal('frontend');
}
