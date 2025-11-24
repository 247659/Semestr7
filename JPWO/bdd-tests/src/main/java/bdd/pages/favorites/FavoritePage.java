package bdd.pages.favorites;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class FavoritePage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By productName = By.cssSelector("[data-test='product-name']");
    private final By removeFromFavoritesButton = By.cssSelector("[data-test='delete']");

    public FavoritePage(WebDriver driver) {
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
        this.driver = driver;
    }

    public void open() {
        driver.get("http://localhost:4200/account/favorites");
    }

    public boolean containsProduct(String expectedName) {
        WebElement nameElement = wait.until(ExpectedConditions.visibilityOfElementLocated(productName));
        return nameElement.getText().contains(expectedName);
    }

    public void removeFromFavorites() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(removeFromFavoritesButton));
        button.click();
    }

}
