import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import pytz


def get_denmark_time():
    """Get the current time in Denmark (CET/CEST)"""
    denmark_tz = pytz.timezone('Europe/Copenhagen')
    denmark_time = datetime.now(denmark_tz)
    return denmark_time


def get_utc_time():
    """Get current UTC time"""
    return datetime.now(pytz.UTC)


def get_imbalance_prices(limit=30):
    """
    Fetch imbalance prices from Energi Data Service API
    """
    try:
        url = f'https://api.energidataservice.dk/dataset/ImbalancePrice?limit={limit}'

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        records = data.get('records', [])

        if not records:
            print("No records found in the response")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Convert time columns to datetime
        if 'TimeUTC' in df.columns:
            df['TimeUTC'] = pd.to_datetime(df['TimeUTC'])
        if 'TimeDK' in df.columns:
            df['TimeDK'] = pd.to_datetime(df['TimeDK'])

        # Sort by UTC time descending (most recent first)
        df = df.sort_values('TimeUTC', ascending=False)

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def get_next_future_times(current_time_dk):
    """
    Calculate the next 5 fixed interval times (00, 15, 30, 45 minutes)
    Based on Denmark time (CET/CEST)
    """
    future_times = []

    # Get current minute in Denmark time
    minute = current_time_dk.minute

    # Calculate the next 15-minute interval
    if minute < 15:
        next_minute = 15
    elif minute < 30:
        next_minute = 30
    elif minute < 45:
        next_minute = 45
    else:
        next_minute = 0
        current_time_dk = current_time_dk + timedelta(hours=1)

    # Create the first future time
    next_time = current_time_dk.replace(minute=next_minute, second=0, microsecond=0)

    # If the calculated time is in the past, add 15 minutes
    if next_time <= current_time_dk:
        next_time = next_time + timedelta(minutes=15)

    # Generate the next 5 times (Denmark time)
    for i in range(5):
        future_times.append(next_time + timedelta(minutes=i * 15))

    return future_times


def check_price_availability(df, target_time_dk):
    """
    Check if a price is available for a specific Denmark time
    Returns the price data if available, None otherwise
    """
    if df is None or df.empty:
        return None

    # Round target time to nearest minute
    target_time_dk = target_time_dk.replace(second=0, microsecond=0)

    # First try to match by TimeDK
    if 'TimeDK' in df.columns:
        for idx, row in df.iterrows():
            if pd.notna(row['TimeDK']):
                if isinstance(row['TimeDK'], pd.Timestamp):
                    record_time_dk = row['TimeDK'].replace(second=0, microsecond=0)
                    if record_time_dk == target_time_dk:
                        return row

    # If not found, try matching by UTC (Denmark = UTC+2)
    if 'TimeUTC' in df.columns:
        target_utc = target_time_dk - timedelta(hours=2)
        for idx, row in df.iterrows():
            if pd.notna(row['TimeUTC']):
                record_utc = row['TimeUTC'].replace(second=0, microsecond=0)
                if record_utc == target_utc:
                    return row

    return None


def get_future_prices():
    """
    Get the next 5 future imbalance prices
    """
    # Get current Denmark time
    current_time_dk = get_denmark_time()
    current_time_utc = get_utc_time()

    print(f"\n🔮 FETCHING FUTURE IMBALANCE PRICES")
    print(f"Current Denmark Time: {current_time_dk.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current UTC Time:     {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Calculate the next 5 fixed interval times
    future_times_dk = get_next_future_times(current_time_dk)

    print("\n📅 NEXT 5 DATA PUBLICATION TIMES (Denmark time):")
    print("-" * 80)
    for i, future_time in enumerate(future_times_dk, 1):
        utc_time = future_time - timedelta(hours=2)
        print(f"  {i}. {future_time.strftime('%Y-%m-%d %H:%M')} (Denmark time)")
        print(f"     UTC: {utc_time.strftime('%Y-%m-%d %H:%M')}Z")

    # Fetch the latest data
    df = get_imbalance_prices(limit=50)

    if df is None or df.empty:
        print("\n❌ No data available from API")
        return None

    print("\n📊 FUTURE IMBALANCE PRICES:")
    print("=" * 80)

    future_prices_found = 0
    future_prices_data = []

    for i, future_time_dk in enumerate(future_times_dk, 1):
        print(f"\n📊 Future Price {i}:")
        print(f"  Denmark Time: {future_time_dk.strftime('%Y-%m-%d %H:%M')}")

        # Check if price is available
        price_data = check_price_availability(df, future_time_dk)

        if price_data is not None:
            print(f"  ✅ Price Available!")

            # Get price area
            price_area = price_data.get('PriceArea', 'N/A')
            print(f"  Price Area: {price_area}")

            # Get imbalance prices
            price_eur = price_data.get('ImbalancePriceEUR', None)
            price_dkk = price_data.get('ImbalancePriceDKK', None)

            if price_eur is not None and pd.notna(price_eur):
                print(f"  Imbalance Price (EUR): €{price_eur:.2f}")
            else:
                print(f"  Imbalance Price (EUR): Not available for this area")

            if price_dkk is not None and pd.notna(price_dkk):
                print(f"  Imbalance Price (DKK): DKK{price_dkk:.2f}")

            # Get additional info
            if 'SatisfiedDemand' in price_data and pd.notna(price_data['SatisfiedDemand']):
                print(f"  Satisfied Demand: {price_data['SatisfiedDemand']:.0f} MW")

            if 'DominatingDirection' in price_data and pd.notna(price_data['DominatingDirection']):
                direction_map = {-1: 'Down', 0: 'Balanced', 1: 'Up'}
                direction = direction_map.get(price_data['DominatingDirection'], str(price_data['DominatingDirection']))
                print(f"  Dominating Direction: {direction}")

            future_prices_found += 1
            future_prices_data.append({
                'time': future_time_dk,
                'price_eur': price_eur,
                'price_dkk': price_dkk,
                'area': price_area,
                'data': price_data
            })
        else:
            print(f"  ⏳ Not yet available")
            # Show when it will be published (approximately 15 minutes after the time)
            publish_time = future_time_dk + timedelta(minutes=15)
            print(f"  Expected publication: {publish_time.strftime('%Y-%m-%d %H:%M')} (Denmark time)")

    print("\n" + "=" * 80)
    print(f"✅ Found {future_prices_found} out of 5 future prices available")

    # Display summary
    if future_prices_data:
        print("\n📋 SUMMARY OF AVAILABLE FUTURE PRICES:")
        print("-" * 80)
        for item in future_prices_data:
            time_str = item['time'].strftime('%Y-%m-%d %H:%M')
            price_str = f"€{item['price_eur']:.2f}" if item['price_eur'] is not None and pd.notna(
                item['price_eur']) else "N/A"
            print(f"  {time_str} (Denmark time): {price_str} | Area: {item['area']}")

    return df


def check_for_new_prices():
    """
    Check if any new prices have arrived
    """
    current_time_dk = get_denmark_time()
    current_time_utc = get_utc_time()

    print(f"\n🔄 Checking for new prices at {current_time_dk.strftime('%Y-%m-%d %H:%M:%S')} (Denmark time)")
    print(f"   UTC Time: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Get the next 5 future times
    future_times_dk = get_next_future_times(current_time_dk)

    # Fetch the latest data
    df = get_imbalance_prices(limit=50)

    if df is None or df.empty:
        print("❌ No data available from API")
        return

    print("\n📊 AVAILABILITY STATUS FOR FUTURE TIME SLOTS:")
    print("-" * 80)

    for i, future_time_dk in enumerate(future_times_dk, 1):
        # Check if price is available
        price_data = check_price_availability(df, future_time_dk)

        status = "✅ AVAILABLE" if price_data is not None else "⏳ WAITING"
        print(f"  {i}. {future_time_dk.strftime('%H:%M')} (Denmark time): {status}")

        if price_data is not None:
            price_eur = price_data.get('ImbalancePriceEUR', None)
            if price_eur is not None and pd.notna(price_eur):
                print(f"     Price: €{price_eur:.2f}")
            area = price_data.get('PriceArea', 'N/A')
            print(f"     Area: {area}")

    print("=" * 80)


def get_detailed_availability():
    """
    Get detailed availability information including all price areas
    """
    print("\n📋 DETAILED AVAILABILITY CHECK:")
    print("=" * 80)

    current_time_dk = get_denmark_time()
    print(f"Current Denmark Time: {current_time_dk.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    # Get next 5 future times
    future_times_dk = get_next_future_times(current_time_dk)

    # Fetch data
    df = get_imbalance_prices(limit=50)

    if df is None or df.empty:
        print("❌ No data available")
        return

    print("\nChecking availability for each time slot across all price areas:\n")

    for i, future_time_dk in enumerate(future_times_dk, 1):
        print(f"Time Slot {i}: {future_time_dk.strftime('%Y-%m-%d %H:%M')} (Denmark time)")
        print("-" * 40)

        # Check all records for this time
        found = False
        if 'TimeDK' in df.columns:
            target_time = future_time_dk.replace(second=0, microsecond=0)

            for idx, row in df.iterrows():
                if pd.notna(row['TimeDK']):
                    if isinstance(row['TimeDK'], pd.Timestamp):
                        record_time = row['TimeDK'].replace(second=0, microsecond=0)
                        if record_time == target_time:
                            area = row.get('PriceArea', 'N/A')
                            price_eur = row.get('ImbalancePriceEUR', None)

                            if price_eur is not None and pd.notna(price_eur):
                                print(f"  ✅ Area {area}: €{price_eur:.2f}")
                            else:
                                print(f"  ⚠️ Area {area}: Data available but price is NaN")
                            found = True

        if not found:
            print(f"  ⏳ No data available yet")

        print()


def run_scheduled_monitoring():
    """
    Run the monitoring at exact 15-minute intervals
    """
    current_time_dk = get_denmark_time()

    # Calculate seconds until the next 15-minute mark
    minute = current_time_dk.minute
    second = current_time_dk.second

    if minute % 15 == 0 and second == 0:
        wait_seconds = 15 * 60
    else:
        next_minute = ((minute // 15) + 1) * 15
        if next_minute >= 60:
            next_minute = 0
            wait_minutes = (60 - minute) + next_minute
        else:
            wait_minutes = next_minute - minute

        if second > 0:
            wait_minutes = wait_minutes - 1
            wait_seconds = (wait_minutes * 60) + (60 - second)
        else:
            wait_seconds = wait_minutes * 60

    next_run = current_time_dk + timedelta(seconds=wait_seconds)
    print(f"\n⏰ Next scheduled run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')} (Denmark time)")
    print(f"Waiting {wait_seconds} seconds...")

    time.sleep(wait_seconds)

    # Check for new prices
    check_for_new_prices()
    get_future_prices()


def main():
    """
    Main function to run the continuous monitoring
    """
    print("=" * 80)
    print("ENERGI DATA SERVICE - FUTURE IMBALANCE PRICE MONITOR")
    print("=" * 80)
    print("This script monitors future imbalance prices at 15-minute intervals")
    print("=" * 80)

    current_time_dk = get_denmark_time()
    current_time_utc = get_utc_time()
    print(f"\n🕐 Current Denmark Time: {current_time_dk.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Current UTC Time:     {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # Show detailed availability
    get_detailed_availability()

    # Initial data fetch
    print("\n📊 INITIAL DATA CHECK:")
    get_future_prices()

    print("\n" + "=" * 80)
    print("Starting continuous monitoring...")
    print("The script will check for new prices at every 15-minute interval")
    print("Press Ctrl+C to stop")
    print("=" * 80)

    try:
        while True:
            run_scheduled_monitoring()
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n👋 Script stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


def test_single_fetch():
    """
    Test function to fetch data once
    """
    print("=" * 80)
    print("TEST - SINGLE DATA FETCH (Using Denmark Time)")
    print("=" * 80)

    current_time_dk = get_denmark_time()
    current_time_utc = get_utc_time()

    print(f"\nDenmark Time: {current_time_dk.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"UTC Time:     {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    future_times_dk = get_next_future_times(current_time_dk)
    print("\nNext 5 future times (Denmark time):")
    for i, future_time in enumerate(future_times_dk, 1):
        utc_time = future_time - timedelta(hours=2)
        print(f"  {i}. {future_time.strftime('%H:%M')}")
        print(f"     UTC: {utc_time.strftime('%H:%M')}Z")

    df = get_imbalance_prices(limit=50)

    if df is not None and not df.empty:
        print("\n📊 Latest data from API:")
        print("-" * 80)
        print("Data available (showing most recent with prices):")

        count = 0
        for idx, row in df.iterrows():
            if count >= 15:
                break

            price_eur = row.get('ImbalancePriceEUR', None)
            if price_eur is not None and pd.notna(price_eur):
                if 'TimeUTC' in row:
                    utc_time = row['TimeUTC']
                    print(f"\nTime UTC: {utc_time.strftime('%Y-%m-%d %H:%M')}Z")
                if 'TimeDK' in row and pd.notna(row['TimeDK']):
                    dk_time = row['TimeDK']
                    print(f"Time DK:  {dk_time.strftime('%Y-%m-%d %H:%M')}")
                if 'PriceArea' in row:
                    print(f"Area: {row['PriceArea']}")
                print(f"Price: €{price_eur:.2f}")
                count += 1

        print("\n" + "=" * 80)
        print("Checking which future prices are available:")
        get_future_prices()

        # Show detailed availability
        get_detailed_availability()
    else:
        print("❌ No data retrieved")


if __name__ == "__main__":
    # Run test first
    test_single_fetch()

    # Uncomment to run continuous monitoring
    main()