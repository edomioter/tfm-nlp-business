# agent/graph_context_provider.py
from typing import Dict, Any, List
from llama_index.graph_stores.neo4j import Neo4jGraphStore


class GraphContextProvider:
    """
    Proveedor de contexto extraído del grafo Neo4j.
    Recupera estadísticas y metadatos relevantes para enriquecer
    la generación de consultas Cypher.
    """

    def __init__(self, graph_store: Neo4jGraphStore):
        self.graph_store = graph_store
        self._cache = {}
        self._cache_ttl = 300  # 5 minutos de caché

    def get_business_context(self, query_str: str) -> Dict[str, Any]:
        """
        Recupera contexto relevante del grafo según la pregunta del usuario.

        Args:
            query_str: Pregunta del usuario en lenguaje natural

        Returns:
            Diccionario con estadísticas y contexto del negocio
        """
        context = {
            "stats": self._get_general_stats(),
            "temporal": self._get_temporal_context(),
            "sources": self._get_top_sources(),
        }

        # Agregar contexto específico según palabras clave
        query_lower = query_str.lower()

        if any(word in query_lower for word in ["ciudad", "ubicación", "dónde", "donde"]):
            context["cities"] = self._get_city_distribution()

        if any(word in query_lower for word in ["conversión", "conversion", "venta", "compra"]):
            context["conversion_metrics"] = self._get_conversion_metrics()

        if any(word in query_lower for word in ["dispositivo", "móvil", "mobile", "desktop"]):
            context["devices"] = self._get_device_distribution()

        return context

    def _get_general_stats(self) -> Dict[str, int]:
        """Obtiene estadísticas generales del grafo."""
        cypher = """
        MATCH (u:User) WITH count(u) as total_users
        MATCH (v:Visitor) WITH total_users, count(v) as total_visitors
        MATCH (s:Session) WITH total_users, total_visitors, count(s) as total_sessions
        MATCH (e:Event) WITH total_users, total_visitors, total_sessions, count(e) as total_events
        MATCH (c:Conversion) WITH total_users, total_visitors, total_sessions, total_events, count(c) as total_conversions
        RETURN total_users, total_visitors, total_sessions, total_events, total_conversions
        """

        try:
            result = self.graph_store.query(cypher)
            if result and len(result) > 0:
                return result[0]
            return {}
        except Exception as e:
            print(f"[GraphContext] Error obteniendo stats generales: {e}")
            return {}

    def _get_temporal_context(self) -> Dict[str, Any]:
        """Obtiene el rango temporal de los datos disponibles."""
        cypher = """
        MATCH (s:Session)
        WHERE s.date IS NOT NULL
        WITH s.date as dates
        RETURN
            min(dates) as min_date,
            max(dates) as max_date,
            count(dates) as sessions_with_date
        """

        try:
            result = self.graph_store.query(cypher)
            if result and len(result) > 0:
                return result[0]
            return {}
        except Exception as e:
            print(f"[GraphContext] Error obteniendo contexto temporal: {e}")
            return {}

    def _get_top_sources(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Obtiene las principales fuentes de tráfico."""
        cypher = f"""
        MATCH (s:Session)
        WHERE s.source IS NOT NULL
        WITH s.source as source, count(s) as session_count
        ORDER BY session_count DESC
        LIMIT {limit}
        RETURN source, session_count
        """

        try:
            result = self.graph_store.query(cypher)
            return result if result else []
        except Exception as e:
            print(f"[GraphContext] Error obteniendo fuentes: {e}")
            return []

    def _get_city_distribution(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Obtiene la distribución de usuarios por ciudad."""
        cypher = f"""
        MATCH (u:User)
        WHERE u.city IS NOT NULL
        WITH u.city as city, count(u) as user_count
        ORDER BY user_count DESC
        LIMIT {limit}
        RETURN city, user_count
        """

        try:
            result = self.graph_store.query(cypher)
            return result if result else []
        except Exception as e:
            print(f"[GraphContext] Error obteniendo ciudades: {e}")
            return []

    def _get_conversion_metrics(self) -> Dict[str, Any]:
        """Calcula métricas clave de conversión."""
        cypher = """
        MATCH (s:Session)
        WITH count(s) as total_sessions
        MATCH (c:Conversion)
        WITH total_sessions, count(c) as total_conversions
        RETURN
            total_sessions,
            total_conversions,
            round(toFloat(total_conversions) / total_sessions * 100, 2) as conversion_rate
        """

        try:
            result = self.graph_store.query(cypher)
            if result and len(result) > 0:
                return result[0]
            return {}
        except Exception as e:
            print(f"[GraphContext] Error calculando métricas de conversión: {e}")
            return {}

    def _get_device_distribution(self) -> List[Dict[str, Any]]:
        """Obtiene la distribución de sesiones por dispositivo."""
        cypher = """
        MATCH (s:Session)
        WHERE s.device IS NOT NULL
        WITH s.device as device, count(s) as session_count
        ORDER BY session_count DESC
        RETURN device, session_count
        """

        try:
            result = self.graph_store.query(cypher)
            return result if result else []
        except Exception as e:
            print(f"[GraphContext] Error obteniendo dispositivos: {e}")
            return []

    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Formatea el contexto del grafo en un string legible para el prompt.

        Args:
            context: Diccionario con el contexto recuperado

        Returns:
            String formateado para incluir en el prompt
        """
        lines = ["--- CONTEXTO DEL NEGOCIO (Datos actuales del grafo) ---"]

        # Stats generales
        if "stats" in context and context["stats"]:
            stats = context["stats"]
            lines.append(f"Total Usuarios Registrados: {stats.get('total_users', 'N/A')}")
            lines.append(f"Total Visitantes: {stats.get('total_visitors', 'N/A')}")
            lines.append(f"Total Sesiones: {stats.get('total_sessions', 'N/A')}")
            lines.append(f"Total Eventos: {stats.get('total_events', 'N/A')}")
            lines.append(f"Total Conversiones: {stats.get('total_conversions', 'N/A')}")

        # Contexto temporal
        if "temporal" in context and context["temporal"]:
            temp = context["temporal"]
            lines.append(f"\nRango de Fechas Disponibles: {temp.get('min_date', 'N/A')} a {temp.get('max_date', 'N/A')}")

        # Fuentes principales
        if "sources" in context and context["sources"]:
            lines.append("\nPrincipales Fuentes de Tráfico:")
            for src in context["sources"]:
                lines.append(f"  - {src.get('source', 'unknown')}: {src.get('session_count', 0)} sesiones")

        # Ciudades
        if "cities" in context and context["cities"]:
            lines.append("\nPrincipales Ciudades:")
            for city in context["cities"]:
                lines.append(f"  - {city.get('city', 'unknown')}: {city.get('user_count', 0)} usuarios")

        # Métricas de conversión
        if "conversion_metrics" in context and context["conversion_metrics"]:
            metrics = context["conversion_metrics"]
            lines.append(f"\nTasa de Conversión Global: {metrics.get('conversion_rate', 'N/A')}%")

        # Dispositivos
        if "devices" in context and context["devices"]:
            lines.append("\nDistribución por Dispositivo:")
            for device in context["devices"]:
                lines.append(f"  - {device.get('device', 'unknown')}: {device.get('session_count', 0)} sesiones")

        lines.append("-----------------------------------------------------------")

        return "\n".join(lines)
