package com.ui.pages.brand;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class BrandAdminPage {
    private WebDriver driver;
    private WebDriverWait wait;

    private By searchField = By.cssSelector("[data-test='brand-search-query']");
    private By searchButton = By.cssSelector("[data-test='brand-search-submit']");


    public BrandAdminPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, java.time.Duration.ofSeconds(10));
    }

    public void open() {
        driver.get("http://localhost:4200/admin/brands");
    }

    public void search(String query) {
        WebElement searchInput = wait.until(ExpectedConditions.visibilityOfElementLocated(searchField));
        searchInput.clear();
        searchInput.sendKeys(query);
        searchInput.submit();
        driver.findElement(searchButton).click();

        wait.until(ExpectedConditions.presenceOfElementLocated(
                By.xpath(String.format("//tr[td[contains(text(),'%s')]]", query))
        ));
    }

    public String getBrandName(String brandName) {
        String xpath = String.format("//tr[td[normalize-space(text())='%s']]/td[2]", brandName);
        return driver.findElement(By.xpath(xpath)).getText();
    }

    public String getBrandSlug(String slug) {
        String xpath = String.format("//tr[td[normalize-space(text())='%s']]/td[3]", slug);
        return driver.findElement(By.xpath(xpath)).getText();
    }
}
