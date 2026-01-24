import { Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard';
import { AlertsPageComponent } from './pages/alerts-page/alerts-page';
import { MapsPageComponent } from './pages/maps-page/maps-page';
import { KnowledgePageComponent } from './pages/knowledge-page/knowledge-page';
import { AboutPageComponent } from './pages/about-page/about-page';

export const routes: Routes = [
  { path: '', component: DashboardComponent }, // Home
  { path: 'alerts', component: AlertsPageComponent },
  { path: 'maps', component: MapsPageComponent },
  { path: 'knowledge', component: KnowledgePageComponent },
  { path: 'about', component: AboutPageComponent },
  { path: '**', redirectTo: '' } // Fallback for 404
];